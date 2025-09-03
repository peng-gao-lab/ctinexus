import os
import time
import logging
from collections import defaultdict

import litellm
import networkx as nx
from omegaconf import DictConfig
from pyvis.network import Network
from scipy.spatial.distance import cosine

from ctinexus.llm_processor import LLMLinker, UsageCalculator
from ctinexus.utils.http_server_utils import get_current_port

logger = logging.getLogger(__name__)


class Linker:
    def __init__(self, config: DictConfig):
        self.config = config
        self.graph = {}

    def call(self, result: dict) -> dict:
        self.js = result
        self.aligned_triplets = self.js["EA"]["aligned_triplets"]

        for triplet in self.aligned_triplets:
            subject_entity_id = triplet["subject"]["entity_id"]
            object_entity_id = triplet["object"]["entity_id"]

            if subject_entity_id not in self.graph:
                self.graph[subject_entity_id] = []
            if object_entity_id not in self.graph:
                self.graph[object_entity_id] = []

            self.graph[subject_entity_id].append(object_entity_id)
            self.graph[object_entity_id].append(subject_entity_id)

        self.subgraphs = self.find_disconnected_subgraphs()
        self.main_nodes = []

        for i, subgraph in enumerate(self.subgraphs):
            main_node_entity_id = self.get_main_node(subgraph)
            main_node = self.get_node(main_node_entity_id)
            logger.info(f"subgraph {i}: main node: {main_node['entity_text']}")
            self.main_nodes.append(main_node)

        self.topic_node = self.get_topic_node(self.subgraphs)
        self.main_nodes = [
            node
            for node in self.main_nodes
            if node["entity_id"] != self.topic_node["entity_id"]
        ]
        self.js["LP"] = LLMLinker(self).link()
        self.js["LP"]["topic_node"] = self.topic_node
        self.js["LP"]["main_nodes"] = self.main_nodes
        self.js["LP"]["subgraphs"] = [list(subgraph) for subgraph in self.subgraphs]
        self.js["LP"]["subgraph_num"] = len(self.subgraphs)

        return self.js

    def find_disconnected_subgraphs(self):
        self.visited = set()
        subgraphs = []

        for start_node in self.graph.keys():
            if start_node not in self.visited:
                current_subgraph = set()
                self.dfs_collect(start_node, current_subgraph)
                subgraphs.append(current_subgraph)

        return subgraphs

    def dfs_collect(self, node, current_subgraph):
        if node in self.visited:
            return

        self.visited.add(node)
        current_subgraph.add(node)

        for neighbour in self.graph[node]:
            self.dfs_collect(neighbour, current_subgraph)

    def get_main_node(self, subgraph):
        outdegrees = defaultdict(int)
        self.directed_graph = {}

        for triplet in self.aligned_triplets:
            subject_entity_id = triplet["subject"]["entity_id"]
            object_entity_id = triplet["object"]["entity_id"]

            if subject_entity_id not in self.directed_graph:
                self.directed_graph[subject_entity_id] = []

            self.directed_graph[subject_entity_id].append(object_entity_id)
            outdegrees[subject_entity_id] += 1
            outdegrees[object_entity_id] += 1

        max_outdegree = 0
        main_node = None

        for node in subgraph:
            if outdegrees[node] > max_outdegree:
                max_outdegree = outdegrees[node]
                main_node = node

        return main_node

    def get_node(self, entity_id):
        for triplet in self.aligned_triplets:
            for key, node in triplet.items():
                if key in ["subject", "object"]:
                    if node["entity_id"] == entity_id:
                        return node

    def get_topic_node(self, subgraphs):
        if not subgraphs:
            return {
                "entity_id": -1,
                "entity_text": "",
                "mention_text": "",
                "mention_class": "default",
                "mention_merged": []
            }
            
        max_node_num = 0
        main_subgraph = subgraphs[0]

        for subgraph in subgraphs:
            if len(subgraph) > max_node_num:
                max_node_num = len(subgraph)
                main_subgraph = subgraph

        return self.get_node(self.get_main_node(main_subgraph))


class Merger:
    def __init__(self, config: DictConfig):
        self.config = config
        self.node_dict = {}  # key is mention_id, value is a list of nodes
        self.class_dict = {}  # key is mention_class, value is a set of mention_ids
        self.entity_dict = {}  # key is entity_id, value is a set of mention_ids
        self.emb_dict = {}  # key is mention_id, value is the embedding of the mention
        self.entity_id = 0
        self.usage = {}
        self.response_time = 0
        self.response = {}

    def get_embeddings(self, texts):
        """Get embeddings for multiple texts in a single API call"""
        startTime = time.time()
        
        api_base_url = None
        provider = self.config.provider.lower()
        embedding_model = self.config.embedding_model

        if provider == "gemini":
            embedding_model = f"gemini/{embedding_model}"
        elif provider == "ollama":
            api_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            embedding_model = f"ollama/{embedding_model}"

        self.response = litellm.embedding(
            model=embedding_model,
            input=texts,
            api_base=api_base_url
        )
        
        self.usage = UsageCalculator(self.config, self.response).calculate()
        self.response_time = time.time() - startTime
        return [item["embedding"] for item in self.response["data"]]

    def calculate_similarity(self, node1, node2):
        """Calculate the cosine similarity between two nodes based on their embeddings."""

        emb1 = self.emb_dict[node1]
        emb2 = self.emb_dict[node2]
        # Calculate cosine similarity
        similarity = 1 - cosine(emb1, emb2)
        return similarity

    def get_entity_text(self, cluster):
        m_freq = {}  # key is mention_id, value is the frequency of the mention
        for m_id in cluster:
            m_freq[m_id] = len(self.node_dict[m_id])
        # sort the mention_id by frequency
        sorted_m_freq = sorted(m_freq.items(), key=lambda x: x[1], reverse=True)
        # get the mention_id with the highest frequency
        mention_id = sorted_m_freq[0][0]
        # get the mention_text of the mention_id
        mention_text = self.node_dict[mention_id][0]["mention_text"]
        return mention_text

    def retrieve_node_list(self, m_id) -> list:
        """Retrieve the list of nodes with the given mention_id from the JSON data."""

        if m_id in self.node_dict:
            return self.node_dict[m_id]

        else:
            raise ValueError(f"Node with mention_id {m_id} not found in the JSON data.")

    def update_class_dict(self, node):
        """Update the class dictionary with the mention class and its mention_id."""

        if node["mention_class"] not in self.class_dict:
            self.class_dict[node["mention_class"]] = set()

        self.class_dict[node["mention_class"]].add(node["mention_id"])

    def call(self, result: dict) -> dict:
        self.js = result

        for triple in self.js["EA"]["aligned_triplets"]:
            for key, node in triple.items():
                if key in ["subject", "object"]:
                    if node["mention_id"] not in self.node_dict:
                        self.node_dict[node["mention_id"]] = []
                    self.node_dict[node["mention_id"]].append(node)

        texts_to_embed = []
        mention_ids = []
        for key, node_list in self.node_dict.items():
            if key not in self.emb_dict:
                texts_to_embed.append(node_list[0]["mention_text"])
                mention_ids.append(key)

        # Get embeddings in batch if there are texts to embed
        if texts_to_embed:
            embeddings = self.get_embeddings(texts_to_embed)
            for mention_id, embedding in zip(mention_ids, embeddings):
                self.emb_dict[mention_id] = embedding

        for triple in self.js["EA"]["aligned_triplets"]:
            for key, node in triple.items():
                if key in ["subject", "object"]:
                    self.update_class_dict(node)

        for mention_class, grouped_nodes in self.class_dict.items():
            if len(grouped_nodes) == 1:
                node_list = self.retrieve_node_list(next(iter(grouped_nodes)))

                for node in node_list:
                    node["entity_id"] = self.entity_id
                    node["mention_merged"] = []
                    node["entity_text"] = node["mention_text"]

                self.entity_id += 1

            elif len(grouped_nodes) > 1:
                clusters = {}  # key is mention_id, value is a set of merged mention_ids
                node_pairs = [
                    (node1, node2)
                    for i, node1 in enumerate(grouped_nodes)
                    for node2 in list(grouped_nodes)[i + 1 :]
                ]

                for node1, node2 in node_pairs:
                    if node1 not in clusters:
                        clusters[node1] = set()

                    if node2 not in clusters:
                        clusters[node2] = set()

                    similarity = self.calculate_similarity(node1, node2)

                    if similarity >= self.config.similarity_threshold:
                        clusters[node1].add(node2)
                        clusters[node2].add(node1)

                unique_clusters = []

                for m_id, merged_ids in clusters.items():
                    temp_cluster = set(merged_ids)
                    temp_cluster.add(m_id)

                    if temp_cluster not in unique_clusters:
                        unique_clusters.append(temp_cluster)

                for cluster in unique_clusters:
                    entity_id = self.entity_id
                    self.entity_id += 1
                    entity_text = self.get_entity_text(cluster)
                    mention_merged = [
                        self.node_dict[m_id][0]["mention_text"] for m_id in cluster
                    ]

                    for m_id in cluster:
                        node_list = self.retrieve_node_list(m_id)

                        for node in node_list:
                            node["entity_id"] = entity_id
                            node["mention_merged"] = [
                                m_text
                                for m_text in mention_merged
                                if m_text != node["mention_text"]
                            ]
                            node["entity_text"] = entity_text

        self.js["EA"]["entity_num"] = self.entity_id
        self.js["EA"]["model_usage"] = self.usage
        self.js["EA"]["response_time"] = self.response_time
        return self.js

def create_graph_visualization(result: dict) -> str:
    """Create an interactive graph visualization using Pyvis"""
    # Create a directed graph
    G = nx.DiGraph()

    http_port = get_current_port()

    # Define colors for different entity types
    entity_colors = {
        "Malware": "#ff4444",  # Bright Red
        "Tool": "#44ff44",  # Bright Green
        "Event": "#4444ff",  # Bright Blue
        "Organization": "#ffaa44",  # Bright Orange
        "Time": "#aa44ff",  # Bright Purple
        "Information": "#44ffff",  # Bright Cyan
        "Indicator": "#ff44ff",  # Bright Magenta
        "Indicator:File": "#ff44ff",  # Bright Magenta
        "default": "#aaaaaa",  # Light Gray
    }

    # Add nodes and edges from the aligned triplets
    if "EA" in result and "aligned_triplets" in result["EA"]:
        for triplet in result["EA"]["aligned_triplets"]:
            # Add subject node
            subject = triplet["subject"]
            G.add_node(
                subject["entity_id"],
                text=subject["entity_text"],
                type=subject["mention_class"],
                color=entity_colors.get(
                    subject["mention_class"], entity_colors["default"]
                ),
            )

            # Add object node
            object_node = triplet["object"]
            G.add_node(
                object_node["entity_id"],
                text=object_node["entity_text"],
                type=object_node["mention_class"],
                color=entity_colors.get(
                    object_node["mention_class"], entity_colors["default"]
                ),
            )

            # Add edge with relation
            G.add_edge(
                subject["entity_id"],
                object_node["entity_id"],
                relation=triplet["relation"],
            )

    # Add predicted links if available
    if "LP" in result and "predicted_links" in result["LP"]:
        for link in result["LP"]["predicted_links"]:
            G.add_edge(
                link["subject"]["entity_id"],
                link["object"]["entity_id"],
                relation=link["relation"],
                predicted=True,
            )

    # Create a Pyvis network
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#27272a",
        font_color="white",
        directed=True,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
            "gravitationalConstant": -500,
            "springLength": 200,
            "springConstant": 0,
            "damping": 0.4
        }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic",
          "roundness": 0.5
        },
        "font": {
          "size": 15,
          "color": "#ffffff",
          "strokeWidth": 1,
          "strokeColor": "#000000"
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)

    # Add nodes to pyvis network
    for node_id, node_attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=node_attrs["text"],
            title=f"{node_attrs['text']}",
            color=node_attrs["color"],
            size=15 + len(G[node_id]) * 2,
        )

    # Add edges to pyvis network
    for u, v, edge_attrs in G.edges(data=True):
        net.add_edge(
            u,
            v,
            label=edge_attrs["relation"],
            title=edge_attrs["relation"],
            color="#ff4444" if edge_attrs.get("predicted") else "#666666",
        )

    # Save the graph to the pyvis_files directory
    timestamp = int(time.time() * 1000)
    file_name = f"network_{timestamp}.html"
    output_dir = os.path.join(os.getcwd(), "ctinexus_output")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    try:
        net.save_graph(file_path)

        # Custom HTML/CSS for the legend
        with open(file_path, "r") as f:
            html_content = f.read()

        legend_html = """
        <div style="position: fixed; top: 50px; right: 20px; background-color: #27272a; color: white; padding: 15px; border-radius: 8px; border: 1px solid #444; max-width: 200px; font-size: 15px;">
            <h3 style="margin-top: 0; font-size: 18px;">Legend</h3>
            <h4 style="margin-bottom: 5px; font-size: 15px;">Node Types:</h4>
            <ul style="list-style: none; padding: 0; margin-bottom: 15px;">
        """

        for entity_type, color in entity_colors.items():
            if entity_type != "default":
                legend_html += f"<li style='margin-bottom: 5px;'><span style='display: inline-block; width: 15px; height: 15px; background-color: {color}; margin-right: 10px; border-radius: 50%;'></span>{entity_type}</li>"

        legend_html += """
            </ul>
            <br>
            <h4 style="margin-bottom: 5px; font-size: 15px;">Edge Types:</h4>
            <ul style="list-style: none; padding: 0;">
            <li style='margin-bottom: 5px;'><span style='display: inline-block; width: 20px; height: 2px; background-color: #94a3b8; margin-right: 10px;'></span>Extracted</li>
            <li style='margin-bottom: 5px;'><span style='display: inline-block; width: 20px; height: 3px; background: repeating-linear-gradient(to right, #ff6b6b 0px, #ff6b6b 5px, transparent 5px, transparent 10px); margin-right: 10px;'></span>Predicted</li>
            </ul>
        </div>
        """

        # Inject the legend HTML into the Pyvis graph HTML
        html_content = html_content.replace("</body>", legend_html + "</body>")
        with open(file_path, "w") as f:
            f.write(html_content)

    except Exception as e:
        logger.error(f"Error saving graph: {e}")

    return f"http://localhost:{http_port}/{file_name}", file_path
