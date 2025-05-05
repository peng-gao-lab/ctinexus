from openai import OpenAI
import time
import json
import logging
LOG = logging.getLogger(__name__)
import requests
from json import JSONDecodeError

class LLMCaller:
    
    def __init__(self, config, prompt):
        self.config = config  
        self.prompt = prompt
    
    def query_llama(self):

        OLLAMA_API_URL = "http://localhost:11434/api/chat"

        payload = {
            "model": "llama3:70b",
            "messages": self.prompt,  
            "stream": False,
            "format": "json"
        }

        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            response = requests.post(OLLAMA_API_URL, json=payload)

            try:
                response_text = response.json()["message"]["content"]
                response_text = response_text.replace("\n", "")  
                response_json = json.loads(response_text)
                return response_json
            
            except JSONDecodeError as e:
                LOG.error("JSONDecodeError: Unable to parse the JSON response: %s", e)
            
            except Exception as e:
                LOG.error("Error: An unexpected error occurred: %s", e)

            attempts += 1
            LOG.info("Retrying... (Attempt %d of %d)", attempts, max_retries)
            LOG.info("Response: %s", response.text)

        LOG.error("Maximum retries reached. Exiting...")
        
        return None
    
    def query_qwen(self):

        OLLAMA_API_URL = "http://localhost:11434/api/chat"

        payload = {
            "model": "qwen2.5:72b",
            "messages": self.prompt,  
            "stream": False,
            "format": "json"
        }

        max_retries = 5
        attempts = 0

        while attempts < max_retries:
            response = requests.post(OLLAMA_API_URL, json=payload)

            try:
                response_text = response.json()["message"]["content"]
                response_text = response_text.replace("\n", "") 
                response_json = json.loads(response_text)
                return response_json
            
            except JSONDecodeError as e:
                LOG.error("JSONDecodeError: Unable to parse the JSON response: %s", e)
            
            except Exception as e:
                LOG.error("Error: An unexpected error occurred: %s", e)

            attempts += 1
            LOG.info("Retrying... (Attempt %d of %d)", attempts, max_retries)
            LOG.info("Response: %s", response.text)

        LOG.error("Maximum retries reached. Exiting...")
        
        return None
    
    def call(self):
        startTime = time.time()

        if self.config.model == "LLaMA":
            response = self.query_llama()
            JSONResp = response
            success = True

        elif self.config.model == "QWen":
            response = self.query_qwen()
            JSONResp = response
            success = True
        
        else:
            attempts = 0
            max_attempts = 5
            success = False
            
            while attempts < max_attempts and not success:
                client = OpenAI(api_key=self.config.api_key)
                
                response = client.chat.completions.create(
                    model = self.config.model,
                    response_format = { "type": "json_object" },
                    messages = self.prompt,
                    max_tokens= 4096,
                )
                
                try:
                    JSONResp = json.loads(response.choices[0].message.content)
                    JSONResp["triplets"]
                    success = True
                
                except (json.decoder.JSONDecodeError, KeyError) as e:
                    attempts += 1
                    LOG.error(f"Attempt {attempts}: {str(e)}")
                    
                    if attempts < max_attempts:
                        LOG.info("Retrying...")
                    
                    else:
                        LOG.error("Maximum attempts reached. Failing...")
                        raise e 
                    
        endTime = time.time()
        generation_time = endTime - startTime

        return response, generation_time, JSONResp