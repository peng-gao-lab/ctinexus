import logging
import os
import re
import time
from collections import defaultdict

import litellm
import networkx as nx
from omegaconf import DictConfig
from pyvis.network import Network
from scipy.spatial.distance import cosine

from ctinexus.llm_processor import LLMLinker, UsageCalculator
from ctinexus.utils.http_server_utils import get_current_port

logger = logging.getLogger(__name__)


def validate_aligned_triplet(triplet: dict) -> bool:
	"""Validate that an aligned triplet has the required structure for graph construction.

	A valid aligned triplet must have:
	- 'subject' dict with 'entity_id' and 'entity_text' keys
	- 'relation' string
	- 'object' dict with 'entity_id' and 'entity_text' keys
	"""
	if not isinstance(triplet, dict):
		return False

	required_keys = ["subject", "relation", "object"]
	if not all(key in triplet for key in required_keys):
		return False

	for key in ["subject", "object"]:
		value = triplet.get(key)
		if not isinstance(value, dict):
			return False
		# Must have entity_id (can be 0) and entity_text
		if "entity_id" not in value:
			return False
		if not value.get("entity_text") and not value.get("mention_text"):
			return False

	relation = triplet.get("relation")
	if not isinstance(relation, str) or not relation.strip():
		return False

	return True


class Linker:
	def __init__(self, config: DictConfig):
		self.config = config
		self.graph = {}

	def call(self, result: dict) -> dict:
		self.js = result
		aligned_triplets = self.js.get("EA", {}).get("aligned_triplets", [])

		# Validate and filter triplets
		if not isinstance(aligned_triplets, list):
			logger.warning("[Linker] aligned_triplets is not a list, resetting to empty")
			aligned_triplets = []

		valid_triplets = []
		for i, triplet in enumerate(aligned_triplets):
			if validate_aligned_triplet(triplet):
				valid_triplets.append(triplet)
			else:
				logger.warning(f"[Linker] Dropping invalid triplet at index {i}: {triplet}")

		if len(valid_triplets) < len(aligned_triplets):
			logger.warning(
				f"[Linker] Filtered {len(aligned_triplets) - len(valid_triplets)} invalid triplets, "
				f"{len(valid_triplets)} remaining"
			)

		self.aligned_triplets = valid_triplets
		self.js["EA"]["aligned_triplets"] = valid_triplets

		# Handle empty triplets case
		if not self.aligned_triplets:
			logger.warning("[Linker] No valid triplets to process, returning empty graph")
			self.js["LP"] = {
				"predicted_links": [],
				"response_time": 0,
				"model_usage": {
					"model": self.config.model,
					"input": {"tokens": 0, "cost": 0},
					"output": {"tokens": 0, "cost": 0},
					"total": {"tokens": 0, "cost": 0},
				},
				"topic_node": {
					"entity_id": -1,
					"entity_text": "",
					"mention_text": "",
					"mention_class": "default",
					"mention_merged": [],
				},
				"main_nodes": [],
				"subgraphs": [],
				"subgraph_num": 0,
			}
			return self.js

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
			logger.debug(f"subgraph {i}: main node: {main_node['entity_text']}")
			self.main_nodes.append(main_node)

		self.topic_node = self.get_topic_node(self.subgraphs)
		self.main_nodes = [node for node in self.main_nodes if node["entity_id"] != self.topic_node["entity_id"]]
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
				"mention_merged": [],
			}

		max_node_num = 0
		main_subgraph = subgraphs[0]

		for subgraph in subgraphs:
			if len(subgraph) > max_node_num:
				max_node_num = len(subgraph)
				main_subgraph = subgraph

		return self.get_node(self.get_main_node(main_subgraph))


def validate_preprocessed_triplet(triplet: dict) -> bool:
	"""Validate that a preprocessed triplet has the required structure for merging.

	A valid preprocessed triplet must have:
	- 'subject' dict with 'mention_id', 'mention_text', 'mention_class' keys
	- 'relation' string
	- 'object' dict with 'mention_id', 'mention_text', 'mention_class' keys
	"""
	if not isinstance(triplet, dict):
		return False

	required_keys = ["subject", "relation", "object"]
	if not all(key in triplet for key in required_keys):
		return False

	for key in ["subject", "object"]:
		value = triplet.get(key)
		if not isinstance(value, dict):
			return False
		# Must have mention_id (can be 0), mention_text, and mention_class
		if "mention_id" not in value:
			return False
		if not value.get("mention_text"):
			return False
		if "mention_class" not in value:
			return False

	relation = triplet.get("relation")
	if not isinstance(relation, str) or not relation.strip():
		return False

	return True


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

		self.response = litellm.embedding(model=embedding_model, input=texts, api_base=api_base_url)

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

		# Validate and filter triplets
		aligned_triplets = self.js.get("EA", {}).get("aligned_triplets", [])
		if not isinstance(aligned_triplets, list):
			logger.warning("[Merger] aligned_triplets is not a list, resetting to empty")
			aligned_triplets = []

		valid_triplets = []
		for i, triplet in enumerate(aligned_triplets):
			if validate_preprocessed_triplet(triplet):
				valid_triplets.append(triplet)
			else:
				logger.warning(f"[Merger] Dropping invalid triplet at index {i}: {triplet}")

		if len(valid_triplets) < len(aligned_triplets):
			logger.warning(
				f"[Merger] Filtered {len(aligned_triplets) - len(valid_triplets)} invalid triplets, "
				f"{len(valid_triplets)} remaining"
			)

		self.js["EA"]["aligned_triplets"] = valid_triplets

		# Handle empty triplets case
		if not valid_triplets:
			logger.warning("[Merger] No valid triplets to process")
			self.js["EA"]["entity_num"] = 0
			self.js["EA"]["model_usage"] = {
				"model": self.config.embedding_model,
				"input": {"tokens": 0, "cost": 0},
				"output": {"tokens": 0, "cost": 0},
				"total": {"tokens": 0, "cost": 0},
			}
			self.js["EA"]["response_time"] = 0
			return self.js

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
					(node1, node2) for i, node1 in enumerate(grouped_nodes) for node2 in list(grouped_nodes)[i + 1 :]
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
					mention_merged = [self.node_dict[m_id][0]["mention_text"] for m_id in cluster]

					for m_id in cluster:
						node_list = self.retrieve_node_list(m_id)

						for node in node_list:
							node["entity_id"] = entity_id
							node["mention_merged"] = [
								m_text for m_text in mention_merged if m_text != node["mention_text"]
							]
							node["entity_text"] = entity_text

		self.js["EA"]["entity_num"] = self.entity_id
		self.js["EA"]["model_usage"] = self.usage
		self.js["EA"]["response_time"] = self.response_time
		return self.js


def _strip_type_suffix(text: str) -> str:
	"""Remove parenthetical type suffix from text, e.g., 'ShadowStrike (Malware)' -> 'ShadowStrike'"""
	return re.sub(r"\s*\([^)]+\)\s*$", "", text).strip()


# Visualization styling constants
_VIS_CONFIG = {
	"background_color": "#27272a",
	"font_color": "#ffffff",
	"stroke_color": "#000000",
	"node_font_size": 18,
	"edge_font_size": 16,
	"node_base_size": 20,
	"node_size_multiplier": 3,
	"edge_base_length": 500,
	"edge_label_multiplier": 15,
	"node_label_multiplier": 8,
	"spring_base_length": 500,
	"spring_edge_multiplier": 12,
	"spring_node_multiplier": 10,
	"gravitational_constant": -8000,
	"extracted_edge_color": "#666666",
	"predicted_edge_color": "#ff4444",
}

# Entity type to color mapping
_ENTITY_COLORS = {
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


def create_graph_visualization(result: dict) -> str:
	"""Create an interactive graph visualization using Pyvis"""
	G = nx.DiGraph()

	# Add nodes and edges from the aligned triplets
	if "EA" in result and "aligned_triplets" in result["EA"]:
		for triplet in result["EA"]["aligned_triplets"]:
			# Add subject node
			subject = triplet.get("subject", {})
			G.add_node(
				subject.get("entity_id"),
				text=subject.get("entity_text", ""),
				type=subject.get("mention_class", "default"),
				color=_ENTITY_COLORS.get(subject.get("mention_class", "default"), _ENTITY_COLORS["default"]),
			)

			# Add object node
			object_node = triplet.get("object", {})
			G.add_node(
				object_node.get("entity_id"),
				text=object_node.get("entity_text", ""),
				type=object_node.get("mention_class", "default"),
				color=_ENTITY_COLORS.get(object_node.get("mention_class", "default"), _ENTITY_COLORS["default"]),
			)

			# Add edge with relation
			G.add_edge(
				subject.get("entity_id"),
				object_node.get("entity_id"),
				relation=triplet.get("relation", ""),
			)

	# Add predicted links if available
	if "LP" in result and "predicted_links" in result["LP"]:
		for link in result["LP"]["predicted_links"]:
			G.add_edge(
				link.get("subject", {}).get("entity_id"),
				link.get("object", {}).get("entity_id"),
				relation=link.get("relation", ""),
				predicted=True,
			)

	# Calculate dynamic spring length based on longest labels in the graph
	max_edge_label_len = max((len(d.get("relation", "")) for _, _, d in G.edges(data=True)), default=0)
	max_node_label_len = max((len(_strip_type_suffix(d.get("text", ""))) for _, d in G.nodes(data=True)), default=0)

	cfg = _VIS_CONFIG
	base_spring_length = (
		cfg["spring_base_length"]
		+ max_edge_label_len * cfg["spring_edge_multiplier"]
		+ max_node_label_len * cfg["spring_node_multiplier"]
	)

	# Create Pyvis network
	net = Network(
		height="100vh",
		width="100%",
		bgcolor=cfg["background_color"],
		font_color=cfg["font_color"],
		directed=True,
	)

	net.set_options(f"""
    {{
      "physics": {{
        "enabled": true,
        "barnesHut": {{
            "gravitationalConstant": {cfg["gravitational_constant"]},
            "springLength": {base_spring_length},
            "springConstant": 0,
            "damping": 0.4,
            "avoidOverlap": 1
        }}
      }},
      "nodes": {{
        "font": {{
          "size": {cfg["node_font_size"]},
          "color": "{cfg["font_color"]}",
          "strokeWidth": 2,
          "strokeColor": "{cfg["stroke_color"]}"
        }}
      }},
      "edges": {{
        "smooth": {{
          "enabled": true,
          "type": "curvedCW",
          "roundness": 0.15
        }},
        "font": {{
          "size": {cfg["edge_font_size"]},
          "color": "{cfg["font_color"]}",
          "strokeWidth": 2,
          "strokeColor": "{cfg["stroke_color"]}"
        }}
      }},
      "interaction": {{
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true
      }},
      "layout": {{
        "improvedLayout": true
      }}
    }}
    """)

	# Add nodes to pyvis network
	for node_id, node_attrs in G.nodes(data=True):
		label_text = _strip_type_suffix(node_attrs.get("text", ""))
		net.add_node(
			node_id,
			label=label_text,
			title=label_text,
			color=node_attrs.get("color", _ENTITY_COLORS["default"]),
			size=cfg["node_base_size"] + len(G[node_id]) * cfg["node_size_multiplier"],
			font={
				"size": cfg["node_font_size"],
				"color": cfg["font_color"],
				"strokeWidth": 2,
				"strokeColor": cfg["stroke_color"],
			},
		)

	# Add edges to pyvis network with dynamic length based on labels
	for u, v, edge_attrs in G.edges(data=True):
		relation = edge_attrs.get("relation", "")
		source_text = _strip_type_suffix(G.nodes[u].get("text", "")) if u in G.nodes else ""
		target_text = _strip_type_suffix(G.nodes[v].get("text", "")) if v in G.nodes else ""

		edge_length = (
			cfg["edge_base_length"]
			+ len(relation) * cfg["edge_label_multiplier"]
			+ (len(source_text) + len(target_text)) * cfg["node_label_multiplier"]
		)

		net.add_edge(
			u,
			v,
			label=relation,
			title=relation,
			color=cfg["predicted_edge_color"] if edge_attrs.get("predicted") else cfg["extracted_edge_color"],
			length=edge_length,
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

		legend_html = f"""
        <div style="position: fixed; top: 50px; right: 20px; background-color: {cfg["background_color"]}; color: white; padding: 15px; border-radius: 8px; border: 1px solid #444; max-width: 200px; font-size: 15px;">
            <h3 style="margin-top: 0; font-size: 18px;">Legend</h3>
            <h4 style="margin-bottom: 5px; font-size: 15px;">Node Types:</h4>
            <ul style="list-style: none; padding: 0; margin-bottom: 15px;">
        """

		for entity_type, color in _ENTITY_COLORS.items():
			legend_html += f"<li style='margin-bottom: 5px;'><span style='display: inline-block; width: 15px; height: 15px; background-color: {color}; margin-right: 10px; border-radius: 50%;'></span>{entity_type.capitalize()}</li>"

		legend_html += f"""
            </ul>
            <br>
            <h4 style="margin-bottom: 5px; font-size: 15px;">Edge Types:</h4>
            <ul style="list-style: none; padding: 0;">
            <li style='margin-bottom: 5px;'><span style='display: inline-block; width: 20px; height: 2px; background-color: {cfg["extracted_edge_color"]}; margin-right: 10px;'></span>Extracted</li>
            <li style='margin-bottom: 5px;'><span style='display: inline-block; width: 20px; height: 2px; background-color: {cfg["predicted_edge_color"]}; margin-right: 10px;'></span>Predicted</li>
            </ul>
        </div>
        """

		# Inject the legend HTML into the Pyvis graph HTML
		html_content = html_content.replace("</body>", legend_html + "</body>")
		with open(file_path, "w") as f:
			f.write(html_content)

	except Exception as e:
		logger.error(f"Error saving graph: {e}")

	# Get port (this also ensures server is running and healthy)
	http_port = get_current_port()
	graph_url = f"http://localhost:{http_port}/{file_name}"

	logger.debug(f"Graph saved to {file_path}, accessible at {graph_url}")

	return graph_url, file_path
