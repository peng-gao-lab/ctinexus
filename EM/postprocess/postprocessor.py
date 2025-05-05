import json
import os 
from omegaconf import DictConfig
from detectIOC import IOC_detect

class Postprocessor:

    def __init__(self, config: DictConfig, inFile: str):
        self.config = config
        self.inFile = inFile
        inFilePath = os.path.join(config.inSet, inFile)
        
        with open(inFilePath, 'r') as f:
            self.js = json.load(f)
        
        self.outFile = os.path.join(config.outSet, inFile)
        os.makedirs(os.path.dirname(self.outFile), exist_ok=True)

        self.mention_dict = {} # key is mention_text, value is mention_id
        self.node_dict = {} # key is mention_id, value is a list of nodes
        self.entity_idx = self.js["EA"]["entity_num"]
        
        for triple in self.js["EA"]["aligned_triplets"]:
            
            for key, node in triple.items():
                
                if key in ["subject", "object"]:
                    
                    if node["mention_text"] not in self.mention_dict:
                        self.mention_dict[node["mention_text"]] = node["mention_id"]

                    if node["mention_id"] not in self.node_dict:
                        self.node_dict[node["mention_id"]] = []
                    self.node_dict[node["mention_id"]].append(node) # with reference to the original node


    def postprocess(self):

        for triple in self.js["EA"]["aligned_triplets"]:
            
            for key, node in triple.items():
                
                if key in ["subject", "object"]:
                    
                    if node["mention_merged"] == []:
                        continue

                    else:
                        
                        iocs = IOC_detect(node["mention_merged"], node["mention_text"])

                        if len(iocs):

                            if len(iocs) < len(node["mention_merged"]) + 1:
                                #This means not all the merged mentions are IOC
                                #TODO: Need to call LLM to check if the non-IOC mentions should be merged with the IOC mentions
                                pass
                            
                            else:
                                #This means all the merged mentions are IOC
                                for m_text in iocs:
                                    m_id = self.mention_dict[m_text]
                                    node_list = self.node_dict[m_id]
                                    entity_id = self.entity_idx
                                    
                                    if node_list[0]["entity_text"] != m_text:
                                        # This is the new entity
                                        self.entity_idx += 1
                                        self.js["EA"]["entity_num"] += 1

                                    for node in node_list:

                                        if node["mention_text"] == node["entity_text"]:
                                            # This is the original mention
                                            node["mention_merged"] = []
                                            continue
                                        
                                        else:
                                            node["mention_merged"] = []
                                            node["entity_id"] = entity_id
                                            node["entity_text"] = m_text

        # Save the merged JSON data
        with open(self.outFile, 'w') as f:
            json.dump(self.js, f, indent=4)