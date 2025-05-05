import json
import os 
from omegaconf import DictConfig, OmegaConf
from LLMLinker import LLMLinker
from collections import defaultdict

class Linker:
    
    def __init__(self, config: DictConfig, inFile):
        
        self.config = config
        self.inFile = inFile
        infile_path = os.path.join(self.config.inSet, self.inFile)
        self.graph = {}
        
        with open(infile_path, 'r') as f:
            self.js = json.load(f)
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
        self.main_nodes = [node for node in self.main_nodes if node["entity_id"] != self.topic_node["entity_id"]]
        self.js["LP"] = LLMLinker(self).link()
        self.js["LP"]["topic_node"] = self.topic_node
        self.js["LP"]["main_nodes"] = self.main_nodes
        self.js["LP"]["subgraphs"] = [list(subgraph) for subgraph in self.subgraphs]
        self.js["LP"]["subgraph_num"] = len(self.subgraphs)
        outfolder = self.config.outSet
        os.makedirs(outfolder, exist_ok=True)
        outfile_path = os.path.join(outfolder, self.inFile)

        with open(outfile_path, 'w') as f:
            json.dump(self.js, f, indent=4)

    
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