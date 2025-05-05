import json
class UsageCalculator:
    def __init__(self, response) -> None:
        self.response = response
        self.model = response.model

    def calculate(self):
        
        with open ("tools/menu/menu.json", "r") as f:
            data = json.load(f)

        iprice = data[self.model]["input"]
        oprice = data[self.model]["output"]
        usageDict = {}
        usageDict["model"] = self.model

        usageDict["input"] = {
            "tokens": self.response.usage.prompt_tokens,
            "cost": iprice*self.response.usage.prompt_tokens
        }

        usageDict["output"] = {
            "tokens": self.response.usage.completion_tokens,
            "cost": oprice*self.response.usage.completion_tokens
        }

        usageDict["total"] = {
            "tokens": self.response.usage.prompt_tokens+self.response.usage.completion_tokens,
            "cost": iprice*self.response.usage.prompt_tokens+oprice*self.response.usage.completion_tokens
        }

        return usageDict