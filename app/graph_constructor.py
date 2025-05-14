import json
import os
from collections import defaultdict

from llm_processor import LLMLinker
from omegaconf import DictConfig
from openai import OpenAI
from scipy.spatial.distance import cosine


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
            print(f"subgraph {i}: main node: {main_node['entity_text']}")
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
        max_node_num = 0

        for subgraph in subgraphs:
            if len(subgraph) > max_node_num:
                max_node_num = len(subgraph)
                main_subgraph = subgraph

        return self.get_node(self.get_main_node(main_subgraph))


class Merger:
    def get_embedding(self, text):
        client = OpenAI(api_key=self.config.api_key)
        response = client.embeddings.create(
            input=text, model=self.config.embedding_model
        )

        return response.data[0].embedding

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

    def __init__(self, config: DictConfig):
        self.config = config
        self.node_dict = {}  # key is mention_id, value is a list of nodes
        self.class_dict = {}  # key is mention_class, value is a set of mention_ids
        self.entity_dict = {}  # key is entity_id, value is a set of mention_ids
        self.emb_dict = {}  # key is mention_id, value is the embedding of the mention
        self.entity_id = 0

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

                    self.node_dict[node["mention_id"]].append(
                        node
                    )  # with reference to the original node

        for key, node_list in self.node_dict.items():
            if key not in self.emb_dict:
                self.emb_dict[key] = self.get_embedding(node_list[0]["mention_text"])

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

        return self.js
