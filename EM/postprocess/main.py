
import os 
import hydra
from omegaconf import DictConfig
from postprocessor import Postprocessor

@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(config: DictConfig):
    
    for JSONFile in os.listdir(config.inSet):
        Postprocessor(config, JSONFile).postprocess()

if __name__ == "__main__":
    run()
