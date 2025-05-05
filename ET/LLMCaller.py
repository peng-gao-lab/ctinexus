from openai import OpenAI
import time
from omegaconf import DictConfig

class LLMCaller:

    def __init__(self, config: DictConfig, prompt) -> None:
        
        self.config = config
        self.prompt = prompt

    def call(self):
        
        client = OpenAI(api_key=self.config.api_key)
        startTime = time.time()
        
        response = client.chat.completions.create(
            model = self.config.model,
            response_format = { "type": "json_object" },
            messages = self.prompt,
            max_tokens= 4096,
        )
        
        endTime = time.time()
        response_time = endTime - startTime
        
        return response, response_time