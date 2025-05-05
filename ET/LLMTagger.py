import json
import os 
from omegaconf import DictConfig
from LLMCaller import LLMCaller
from jinja2 import Environment, FileSystemLoader, meta
from usageCalculator import UsageCalculator


class LLMTagger:

    def __init__(self, config: DictConfig):

        self.config = config


    def tag(self, file):

        inFile_path = os.path.join(self.config.inSet, file)
        
        with open(inFile_path, 'r') as f:
            js = json.load(f)
            triples = js["IE"]["triplets"]
             
        self.prompt = self.generate_prompt(triples)
        promptFolder = self.config.tag_prompt_store
        os.makedirs(promptFolder, exist_ok=True)
        self.promptPath = os.path.join(promptFolder, file.split('.')[0] + ".json")
       
        with open(self.promptPath, 'w') as f:
            f.write(json.dumps(self.prompt[0]["content"], indent=4).replace("\\n", "\n").replace('\\"', '\"'))
             
        self.response, self.response_time = LLMCaller(self.config, self.prompt).call()
        self.usage = UsageCalculator(self.response).calculate()
        self.response_content = json.loads(self.response.choices[0].message.content)
        os.makedirs(self.config.outSet, exist_ok=True)
        outfile_path = os.path.join(self.config.outSet, file)

        with open(outfile_path, 'w') as f:
            js["ET"] = {}
            js["ET"]["typed_triplets"] = self.response_content["tagged_triples"]
            js["ET"]["response_time"] = self.response_time
            js["ET"]["tag_prompt"] = self.promptPath
            js["ET"]["model_usage"] = self.usage
            json.dump(js, f, indent=4)
        

    def generate_prompt(self, triples):
            
            env = Environment(loader=FileSystemLoader(self.config.tag_prompt_folder))
            template_file = env.loader.get_source(env, self.config.tag_prompt_file)[0]
            template = env.get_template(self.config.tag_prompt_file)
            vars = meta.find_undeclared_variables(env.parse(template_file))
            
            if vars is not {}: 
                UserPrompt = template.render(triples=triples)
            
            else:
                UserPrompt = template.render()
            
            prompt = [{"role": "user", "content": UserPrompt}]
            
            return prompt
