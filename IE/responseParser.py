import json
from usageCalculator import UsageCalculator
import logging
LOG = logging.getLogger(__name__)

class ResponseParser:
    def __init__(self, llmExtractor) -> None:
        self.llm_response = llmExtractor.llm_response
        self.prompt = llmExtractor.prompt
        self.query = llmExtractor.inFileJSON["text"]
        self.config = llmExtractor.config
        self.JSONResp = llmExtractor.JSONResp

    def parse(self):

        def get_char_before_hyphen(s):
            index = s.find('-')
            if index > 0:
                return s[:index]
            return None
        
        is_gpt = get_char_before_hyphen(self.config.model) == "gpt"

        self.output = {
            "CTI": self.query,
            "IE": self.JSONResp,
            "usage": UsageCalculator(self.llm_response).calculate() if is_gpt else None,
            "prompt": self.prompt,
            "triples_count": len(self.JSONResp["triplets"]),
        }

        return self.output