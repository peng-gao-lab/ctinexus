import os 
import hydra
from omegaconf import DictConfig
from LLMTagger import LLMTagger

@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
        
        for file in os.listdir(config.inSet):
            LLMTagger(config).tag(file)

if __name__ == "__main__":
    run()