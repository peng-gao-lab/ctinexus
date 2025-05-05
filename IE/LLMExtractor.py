from promptConstructor import PromptConstructor
from demoRetriever import DemoRetriever
from LLMcaller import LLMCaller
from responseParser import ResponseParser
import json
import os 
import time

class LLMExtractor:

    def __init__(self, config, inFile):

        self.config = config
        self.inFile = inFile
    

    def extract(self):

        with open(os.path.join(self.config.inSet, self.inFile), 'r') as f:
            self.inFileJSON = json.load(f)

        if self.config.retriever == "fixed":
            self.demos = None
        
        else:
            self.demos, self.demosInfo = DemoRetriever(self).retriveDemo()

        self.inFilename = os.path.splitext(self.inFile)[0]
        self.prompt = PromptConstructor(self).generate_prompt()
        self.llm_response, self.response_time, self.JSONResp = LLMCaller(self.config, self.prompt).call()
        self.output = ResponseParser(self).parse()

        if self.config.model == "LLaMA" or self.config.model == "QWen":
            self.promptID = str(int(round(time.time() * 1000)))
        
        else:
            self.promptID = self.llm_response.id[-3:]

        templ = self.config.templ.split('.')[0]
        self.outPromptFolder = self.config.prompt_store
        os.makedirs(self.outPromptFolder, exist_ok=True)
        self.outPromptFile = os.path.join(self.outPromptFolder, f'{self.inFilename}-{templ}-{self.promptID}.json')
        with open(self.outPromptFile, 'w') as f:
            json_str = json.dumps(self.output["prompt"])
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
        
        os.makedirs(self.config.outSet, exist_ok=True)
        self.outFile = os.path.join(self.config.outSet, f'{self.inFilename}.json')

        with open(self.outFile, 'w') as f:
            outJSON = {}
            outJSON["text"] = self.output["CTI"]
            outJSON["IE"] = {}
            outJSON["IE"]["triplets"] = self.output["IE"]["triplets"]
            outJSON["IE"]["triples_count"] = self.output["triples_count"]
            outJSON["IE"]["cost"] = self.output["usage"]
            outJSON["IE"]["time"] = self.response_time
            outJSON["IE"]["Prompt"] = {}
            outJSON["IE"]["Prompt"]["constructed_prompt"] = self.outPromptFile
            outJSON["IE"]["Prompt"]["prompt_template"] = self.config.templ

            if self.demos is not None:
                outJSON["IE"]["Prompt"]["demo_retriever"]= self.config.retriever.type
                outJSON["IE"]["Prompt"]["demos"] = self.demosInfo
                outJSON["IE"]["Prompt"]["demo_number"] = self.config.shot
                
                if self.config.retriever.type == "kNN":
                    outJSON["IE"]["Prompt"]["permutation"] = self.config.retriever.permutation
            
            else:
                outJSON["IE"]["Prompt"]["demo_retriever"]= self.config.retriever
          
            #write to file
            json.dump(outJSON, f, indent=4)