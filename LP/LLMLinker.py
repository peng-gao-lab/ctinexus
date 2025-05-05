import json
import os 
from jinja2 import Environment, FileSystemLoader, meta
from LLMCaller import LLMCaller
from UsageCalculator import UsageCalculator


class LLMLinker:
    
    def __init__(self, linker):
        
        self.config = linker.config
        self.predicted_triples = []
        self.response_times = []
        self.usages = []
        self.main_nodes = linker.main_nodes
        self.linker = linker
        self.js = linker.js
        self.inFile = linker.inFile
        self.topic_node = linker.topic_node

    
    def link(self):
        
        for main_node in self.main_nodes:
            prompt = self.generate_prompt(main_node)
            llmCaller = LLMCaller(self.config, prompt)
            self.llm_response, self.response_time = llmCaller.call()
            self.usage = UsageCalculator(self.llm_response).calculate()
            self.response_content = json.loads(self.llm_response.choices[0].message.content)
            
            try:
                pred_sub = self.response_content["predicted_triple"]['subject']
                pred_obj = self.response_content["predicted_triple"]['object']
                pred_rel = self.response_content["predicted_triple"]['relation']
            
            except:
                values = list(self.response_content.values())
                pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]

            if pred_sub == main_node["entity_text"] and pred_obj == self.topic_node["entity_text"]:
                new_sub = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"]
                }
                new_obj = self.topic_node

            elif pred_obj == main_node["entity_text"] and pred_sub == self.topic_node["entity_text"]:
                new_sub = self.topic_node
                new_obj = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"]
                }

            else:
                print("Error: The predicted subject and object do not match the unvisited subject and topic entity, the LLM produce hallucination!")
                print(f"Hallucinated file name: {self.inFile}")
                new_sub = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination"
                }
                new_obj = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination"
                }

            self.predicted_triple = {
                "subject": new_sub,
                "relation": pred_rel,
                "object": new_obj
            }
            self.predicted_triples.append(self.predicted_triple)
            self.response_times.append(self.response_time)
            self.usages.append(self.usage)

        LP = {
            "predicted_links": self.predicted_triples,

            "response_time": sum(self.response_times),
            "model": self.config.model,

            "usage": {
                "input": {
                    "tokens": sum([usage["input"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["input"]["cost"] for usage in self.usages])
                },
                "output": {
                    "tokens": sum([usage["output"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["output"]["cost"] for usage in self.usages])
                },
                "total": {
                    "tokens": sum([usage["total"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["total"]["cost"] for usage in self.usages])
                }
            }
        }
        
        return LP


    def generate_prompt(self, main_node):
            
            env = Environment(loader=FileSystemLoader(self.config.link_prompt_folder))
            parsed_template = env.parse(env.loader.get_source(env, self.config.link_prompt_file)[0])
            template = env.get_template(self.config.link_prompt_file)
            variables = meta.find_undeclared_variables(parsed_template)

            if variables is not {}:
                    User_prompt = template.render(main_node=main_node["entity_text"], CTI=self.js["text"], topic_node=self.topic_node["entity_text"])
            
            else:
                User_prompt = template.render()
            
            prompt = [{"role": "user", "content": User_prompt}]
            FolderPath = self.config.link_prompt_set
            os.makedirs(FolderPath, exist_ok=True)
            
            with open(os.path.join(FolderPath, self.inFile.split('.')[0] + ".json"), 'w') as f:
                f.write(json.dumps(User_prompt, indent=4).replace("\\n", "\n").replace('\\"', '\"'))
            
            return prompt