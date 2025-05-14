import json
import logging
import os
import time
from json import JSONDecodeError

import requests
from jinja2 import Environment, FileSystemLoader, meta
from omegaconf import DictConfig
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMTagger:
    def __init__(self, config: DictConfig):
        self.config = config

    def call(self, result: dict) -> dict:
        triples = result["IE"]["triplets"]

        self.prompt = self.generate_prompt(triples)
        self.response, self.response_time, _ = LLMCaller(
            self.config, self.prompt
        ).call()
        self.usage = UsageCalculator(self.response).calculate()
        self.response_content = json.loads(self.response.choices[0].message.content)

        result["ET"] = {}
        result["ET"]["typed_triplets"] = self.response_content["tagged_triples"]
        result["ET"]["response_time"] = self.response_time
        result["ET"]["model_usage"] = self.usage

        return result

    def generate_prompt(self, triples):
        env = Environment(loader=FileSystemLoader(self.config.tag_prompt_folder))
        template_file = env.loader.get_source(env, self.config.tag_prompt_file)[0]
        template = env.get_template(self.config.tag_prompt_file)
        vars = meta.find_undeclared_variables(env.parse(template_file))

        if vars != {}:
            UserPrompt = template.render(triples=triples)

        else:
            UserPrompt = template.render()

        prompt = [{"role": "user", "content": UserPrompt}]

        return prompt


class LLMLinker:
    def __init__(self, linker):
        self.config = linker.config
        self.predicted_triples = []
        self.response_times = []
        self.usages = []
        self.main_nodes = linker.main_nodes
        self.linker = linker
        self.js = linker.js
        self.topic_node = linker.topic_node

    def link(self):
        for main_node in self.main_nodes:
            prompt = self.generate_prompt(main_node)
            llmCaller = LLMCaller(self.config, prompt)
            self.llm_response, self.response_time, _ = llmCaller.call()
            self.usage = UsageCalculator(self.llm_response).calculate()
            self.response_content = json.loads(
                self.llm_response.choices[0].message.content
            )

            try:
                pred_sub = self.response_content["predicted_triple"]["subject"]
                pred_obj = self.response_content["predicted_triple"]["object"]
                pred_rel = self.response_content["predicted_triple"]["relation"]

            except:
                values = list(self.response_content.values())
                pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]

            if (
                pred_sub == main_node["entity_text"]
                and pred_obj == self.topic_node["entity_text"]
            ):
                new_sub = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"],
                }
                new_obj = self.topic_node

            elif (
                pred_obj == main_node["entity_text"]
                and pred_sub == self.topic_node["entity_text"]
            ):
                new_sub = self.topic_node
                new_obj = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"],
                }

            else:
                print(
                    "Error: The predicted subject and object do not match the unvisited subject and topic entity, the LLM produce hallucination!"
                )
                print(f"Hallucinated in text: {self.js['text']}")
                new_sub = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination",
                }
                new_obj = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination",
                }

            self.predicted_triple = {
                "subject": new_sub,
                "relation": pred_rel,
                "object": new_obj,
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
                    "cost": sum([usage["input"]["cost"] for usage in self.usages]),
                },
                "output": {
                    "tokens": sum([usage["output"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["output"]["cost"] for usage in self.usages]),
                },
                "total": {
                    "tokens": sum([usage["total"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["total"]["cost"] for usage in self.usages]),
                },
            },
        }

        return LP

    def generate_prompt(self, main_node):
        env = Environment(loader=FileSystemLoader(self.config.link_prompt_folder))
        parsed_template = env.parse(
            env.loader.get_source(env, self.config.link_prompt_file)[0]
        )
        template = env.get_template(self.config.link_prompt_file)
        variables = meta.find_undeclared_variables(parsed_template)

        if variables != {}:
            User_prompt = template.render(
                main_node=main_node["entity_text"],
                CTI=self.js["text"],
                topic_node=self.topic_node["entity_text"],
            )

        else:
            User_prompt = template.render()

        prompt = [{"role": "user", "content": User_prompt}]

        return prompt


class LLMCaller:
    def __init__(self, config: DictConfig, prompt) -> None:
        self.config = config
        self.prompt = prompt

    def query_llama(self):
        OLLAMA_API_URL = "http://localhost:11434/api/chat"

        payload = {
            "model": "llama3:70b",
            "messages": self.prompt,
            "stream": False,
            "format": "json",
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
                logger.error(
                    "JSONDecodeError: Unable to parse the JSON response: %s", e
                )

            except Exception as e:
                logger.error("Error: An unexpected error occurred: %s", e)

            attempts += 1
            logger.info("Retrying... (Attempt %d of %d)", attempts, max_retries)
            logger.info("Response: %s", response.text)

        logger.error("Maximum retries reached. Exiting...")

        return None

    def query_qwen(self):
        OLLAMA_API_URL = "http://localhost:11434/api/chat"

        payload = {
            "model": "qwen2.5:72b",
            "messages": self.prompt,
            "stream": False,
            "format": "json",
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
                logger.error(
                    "JSONDecodeError: Unable to parse the JSON response: %s", e
                )

            except Exception as e:
                logger.error("Error: An unexpected error occurred: %s", e)

            attempts += 1
            logger.info("Retrying... (Attempt %d of %d)", attempts, max_retries)
            logger.info("Response: %s", response.text)

        logger.error("Maximum retries reached. Exiting...")

        return None

    def call(self, validate=False) -> tuple[dict, float, dict]:
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
            JSONResp = {}
            attempts = 0
            max_attempts = 5
            success = False

            while attempts < max_attempts and not success:
                client = OpenAI(api_key=self.config.api_key)

                response = client.chat.completions.create(
                    model=self.config.model,
                    response_format={"type": "json_object"},
                    messages=self.prompt,
                    max_tokens=4096,
                )

                if validate:
                    try:
                        JSONResp = json.loads(response.choices[0].message.content)
                        JSONResp["triplets"]
                        success = True

                    except (json.decoder.JSONDecodeError, KeyError) as e:
                        attempts += 1
                        logger.error(f"Attempt {attempts}: {str(e)}")

                        if attempts < max_attempts:
                            logger.info("Retrying...")

                        else:
                            logger.error("Maximum attempts reached. Failing...")
                            raise e
                else:
                    success = True

        endTime = time.time()
        generation_time = endTime - startTime

        return response, generation_time, JSONResp


class LLMExtractor:
    def __init__(self, config):
        self.config = config

    def call(self, query: str) -> dict:
        self.query = query

        self.prompt = PromptConstructor(self).generate_prompt()
        self.llm_response, self.response_time, self.JSONResp = LLMCaller(
            self.config, self.prompt
        ).call(validate=True)
        self.output = ResponseParser(self).parse()

        if self.config.model == "LLaMA" or self.config.model == "QWen":
            self.promptID = str(int(round(time.time() * 1000)))
        else:
            self.promptID = self.llm_response.id[-3:]

        outJSON = {}
        outJSON["text"] = self.output["CTI"]
        outJSON["IE"] = {}
        outJSON["IE"]["triplets"] = self.output["IE"]["triplets"]
        outJSON["IE"]["triples_count"] = self.output["triples_count"]
        outJSON["IE"]["cost"] = self.output["usage"]
        outJSON["IE"]["time"] = self.response_time
        outJSON["IE"]["Prompt"] = {}
        outJSON["IE"]["Prompt"]["prompt_template"] = self.config.templ

        return outJSON


class PromptConstructor:
    def __init__(self, llmExtractor):
        self.config = llmExtractor.config
        self.query = llmExtractor.query
        self.templ = self.config.templ

    def generate_prompt(self) -> list[dict]:
        try:
            if not self.config.ie_prompt_set or not os.path.isdir(
                self.config.ie_prompt_set
            ):
                raise ValueError(
                    f"Invalid template directory: {self.config.ie_prompt_set}"
                )

            env = Environment(loader=FileSystemLoader(self.config.ie_prompt_set))
            DymTemplate = self.templ
            template_source = env.loader.get_source(env, DymTemplate)[0]
            parsed_content = env.parse(template_source)
            variables = meta.find_undeclared_variables(parsed_content)
            template = env.get_template(DymTemplate)

            if variables:
                Uprompt = template.render(query=self.query)
            else:
                Uprompt = template.render()

            prompt = [{"role": "user", "content": Uprompt}]
            return prompt

        except Exception as e:
            raise RuntimeError(f"Error generating prompt: {e}")


class ResponseParser:
    def __init__(self, llmExtractor) -> None:
        self.llm_response = llmExtractor.llm_response
        self.prompt = llmExtractor.prompt
        self.config = llmExtractor.config
        self.JSONResp = llmExtractor.JSONResp
        self.query = llmExtractor.query

    def parse(self):
        def get_char_before_hyphen(s):
            index = s.find("-")
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


class UsageCalculator:
    def __init__(self, response) -> None:
        self.response = response
        self.model = response.model

    def calculate(self):
        # import menu
        with open("app/tools/menu/menu.json", "r") as f:
            data = json.load(f)

        iprice = data[self.model]["input"]
        oprice = data[self.model]["output"]
        usageDict = {}
        usageDict["model"] = self.model
        usageDict["input"] = {
            "tokens": self.response.usage.prompt_tokens,
            "cost": iprice * self.response.usage.prompt_tokens,
        }
        usageDict["output"] = {
            "tokens": self.response.usage.completion_tokens,
            "cost": oprice * self.response.usage.completion_tokens,
        }
        usageDict["total"] = {
            "tokens": self.response.usage.prompt_tokens
            + self.response.usage.completion_tokens,
            "cost": iprice * self.response.usage.prompt_tokens
            + oprice * self.response.usage.completion_tokens,
        }

        return usageDict
