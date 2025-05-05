import os
import json
import time
from omegaconf import DictConfig
import copy
import openai
from scipy.spatial.distance import cosine

class Merger:
    
    def get_embedding(self, text):
        
        openai.api_key = self.config.api_key
        response = openai.Embedding.create(
            input=text,
            engine=self.config.embedding_model
        )
        
        return response["data"][0]["embedding"]
    

    def calculate_similarity(self, node1, node2):
        """ Calculate the cosine similarity between two nodes based on their embeddings."""
        
        emb1 = self.emb_dict[node1]
        emb2 = self.emb_dict[node2]
        # Calculate cosine similarity
        similarity = 1 - cosine(emb1, emb2)
        return similarity


    def get_entity_text(self, cluster):
        m_freq = {} #key is mention_id, value is the frequency of the mention
        for m_id in cluster:
            m_freq[m_id] = len(self.node_dict[m_id])
        # sort the mention_id by frequency
        sorted_m_freq = sorted(m_freq.items(), key=lambda x: x[1], reverse=True)
        # get the mention_id with the highest frequency
        mention_id = sorted_m_freq[0][0]
        # get the mention_text of the mention_id
        mention_text = self.node_dict[mention_id][0]["mention_text"]
        return mention_text
        


    def __init__(self, config: DictConfig, inFile: str):
        
        self.config = config
        self.inFile = inFile
        inFile_path = os.path.join(self.config.inSet, self.inFile)
        
        with open(inFile_path, 'r') as f:
            self.js = json.load(f)
        
        self.node_dict = {} # key is mention_id, value is a list of nodes
        self.class_dict = {} # key is mention_class, value is a set of mention_ids
        self.entity_dict = {} # key is entity_id, value is a set of mention_ids
        self.emb_dict = {} # key is mention_id, value is the embedding of the mention
        self.entity_id = 0


        for triple in self.js["EA"]["aligned_triplets"]:
            
            for key, node in triple.items():
                
                if key in ["subject", "object"]:
                    
                    if node["mention_id"] not in self.node_dict:
                        self.node_dict[node["mention_id"]] = []

                    self.node_dict[node["mention_id"]].append(node) # with reference to the original node
        
        for key, node_list in self.node_dict.items():

            if key not in self.emb_dict:
                self.emb_dict[key] = self.get_embedding(node_list[0]["mention_text"])



    def retrieve_node_list(self, m_id)->list:
        """ Retrieve the list of nodes with the given mention_id from the JSON data."""
        
        if m_id in self.node_dict:
            return self.node_dict[m_id]
        
        else:
            raise ValueError(f"Node with mention_id {m_id} not found in the JSON data.")


    def update_class_dict(self, node):
        """ Update the class dictionary with the mention class and its mention_id."""
        
        if node["mention_class"] not in self.class_dict:
            self.class_dict[node["mention_class"]] = set()
        
        self.class_dict[node["mention_class"]].add(node["mention_id"])


    def merge(self):
              
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
                clusters = {} # key is mention_id, value is a set of merged mention_ids
                node_pairs = [(node1, node2) for i, node1 in enumerate(grouped_nodes) for node2 in list(grouped_nodes)[i+1:]]

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
                            node["mention_merged"] = [m_text for m_text in mention_merged if m_text != node["mention_text"]] 
                            node["entity_text"] = entity_text
        
        self.js["EA"]["entity_num"] = self.entity_id

        # Save the merged JSON data
        outFile_path = os.path.join(self.config.outSet, self.inFile)
        os.makedirs(self.config.outSet, exist_ok=True)
        with open(outFile_path, 'w') as f:
            json.dump(self.js, f, indent=4)
                
                            


                        




