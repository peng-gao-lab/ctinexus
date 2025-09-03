import json
import logging
import os
import random
import re
import time
from functools import wraps

import litellm
import nltk
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, meta
from nltk.corpus import stopwords
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from ctinexus.utils.path_utils import resolve_path

logger = logging.getLogger(__name__)

# Download NLTK stopwords if not already present
try:
    stopwords.words("english")
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)

litellm.drop_params = True


def with_retry(max_attempts=5):
    """Decorator to handle retry logic for API calls"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error("Error in attempt %d: %s", attempt + 1, str(e))
                    if attempt < max_attempts - 1:
                        logger.info("Retrying...")
                    else:
                        logger.error("Maximum retries reached. Exiting...")
                        raise e
            return None

        return wrapper

    return decorator


class LLMTagger:
    def __init__(self, config: DictConfig):
        self.config = config

    def call(self, result: dict) -> dict:
        triples = result["IE"]["triplets"]

        self.prompt = self.generate_prompt(triples)
        self.response, self.response_time = LLMCaller(self.config, self.prompt).call()
        self.usage = UsageCalculator(self.config, self.response).calculate()
        self.response_content = extract_json_from_response(
            self.response.choices[0].message.content
        )

        # Safety check: ensure response_content has required keys
        if not self.response_content or not isinstance(self.response_content, dict):
            self.response_content = {"tagged_triples": []}
        
        if "tagged_triples" not in self.response_content:
            # Try alternative key names that models might use
            if "triplets" in self.response_content:
                self.response_content["tagged_triples"] = self.response_content["triplets"]
            else:
                self.response_content["tagged_triples"] = []

        result["ET"] = {}
        result["ET"]["typed_triplets"] = self.response_content["tagged_triples"]
        result["ET"]["response_time"] = self.response_time
        result["ET"]["model_usage"] = self.usage

        return result

    def generate_prompt(self, triples):
        tag_prompt_folder = self.config.tag_prompt_folder
        env = Environment(loader=FileSystemLoader(resolve_path(tag_prompt_folder)))
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
            self.llm_response, self.response_time = llmCaller.call()
            self.usage = UsageCalculator(self.config, self.llm_response).calculate()
            self.response_content = extract_json_from_response(
                self.llm_response.choices[0].message.content
            )

            # Safety check and extract predicted triple information
            if not self.response_content or not isinstance(self.response_content, dict):
                logger.warning("Invalid response from LLM for link prediction")
                pred_sub, pred_rel, pred_obj = "unknown", "unknown", "unknown"
            else:
                try:
                    if "predicted_triple" in self.response_content:
                        pred_sub = self.response_content["predicted_triple"]["subject"]
                        pred_obj = self.response_content["predicted_triple"]["object"]
                        pred_rel = self.response_content["predicted_triple"]["relation"]
                    else:
                        # Try to extract from flat structure or list of values
                        values = list(self.response_content.values())
                        if len(values) >= 3:
                            pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]
                        else:
                            pred_sub, pred_rel, pred_obj = "unknown", "unknown", "unknown"
                except Exception as e:
                    logger.error(f"Error extracting predicted triple: {e}")
                    pred_sub, pred_rel, pred_obj = "unknown", "unknown", "unknown"

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
                logger.error("The predicted subject and object do not match the unvisited subject and topic entity, the LLM produce hallucination!")
                logger.error(f"Hallucinated in text: {self.js['text']}")

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
            "model_usage": {
                "model": self.config.model,
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
        link_prompt_folder = self.config.link_prompt_folder
        env = Environment(loader=FileSystemLoader(resolve_path(link_prompt_folder)))
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
        self.max_tokens = 4096

    @with_retry()
    def query_llm(self):
        """Query LLM using litellm"""
        try:
            provider = self.config.provider.lower()
            model_id = self.config.model

            # Format request based on model type
            if provider == "anthropic":
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in self.prompt
                    if msg["role"] in ["user", "assistant"]
                ]
                response = litellm.completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
            elif provider == "gemini":
                response = litellm.completion(
                    model=f"gemini/{model_id}",
                    messages=[{"role": "user", "content": self.prompt[-1]["content"]}],
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                    response_format={"type": "json_object"},
                )
            elif provider == "meta":
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": self.prompt[-1]["content"]}],
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                )
            elif provider == "ollama":
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                improved_prompt = self.prompt[-1]["content"] + "\n\nIMPORTANT: output should be a valid JSON object with no extra text or description."

                response = litellm.completion(
                    model=f"ollama/{model_id}",
                    messages=[{"role": "user", "content": improved_prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                    api_base=ollama_base_url,
                )
            else:
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": self.prompt[-1]["content"]}],
                    max_tokens=self.max_tokens,
                    temperature=0.8,
                    response_format={"type": "json_object"},
                )

            return response

        except Exception as e:
            logger.error(f"Error invoking LLM {model_id}: {str(e)}")
            raise Exception(f"Error invoking LLM {model_id}: {str(e)}")

    def call(self) -> tuple[dict, float]:
        startTime = time.time()
        response = self.query_llm()
        generation_time = time.time() - startTime
        return response, generation_time


class LLMExtractor:
    def __init__(self, config):
        self.config = config

    def call(self, query: str) -> dict:
        self.query = query

        if self.config.retriever == "fixed":
            self.demos = None
        else:
            self.demos, self.demosInfo = DemoRetriever(self).retriveDemo()

        self.prompt = PromptConstructor(self).generate_prompt()
        self.llm_response, self.response_time = LLMCaller(
            self.config, self.prompt
        ).call()

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
        outJSON["IE"]["model_usage"] = self.output["usage"]
        outJSON["IE"]["response_time"] = self.response_time
        outJSON["IE"]["Prompt"] = {}
        outJSON["IE"]["Prompt"]["prompt_template"] = self.config.ie_templ

        if self.demos is not None:
            outJSON["IE"]["Prompt"]["demo_retriever"] = self.config.retriever.type
            outJSON["IE"]["Prompt"]["demos"] = self.demosInfo
            outJSON["IE"]["Prompt"]["demo_number"] = self.config.shot

            if self.config.retriever.type == "kNN":
                outJSON["IE"]["Prompt"]["permutation"] = (
                    self.config.retriever.permutation
                )
        else:
            outJSON["IE"]["Prompt"]["demo_retriever"] = self.config.retriever

        return outJSON


class PromptConstructor:
    def __init__(self, llmExtractor):
        self.demos = llmExtractor.demos
        self.config = llmExtractor.config
        self.query = llmExtractor.query
        self.templ = self.config.ie_templ

    def generate_prompt(self) -> list[dict]:
        try:
            ie_prompt_set = self.config.ie_prompt_set
            resolved_prompt_set = resolve_path(ie_prompt_set)
            if not resolved_prompt_set or not os.path.isdir(resolved_prompt_set):
                raise ValueError(
                    f"Invalid template directory: {self.config.ie_prompt_set}"
                )

            env = Environment(loader=FileSystemLoader(resolved_prompt_set))
            DymTemplate = self.templ
            template_source = env.loader.get_source(env, DymTemplate)[0]
            parsed_content = env.parse(template_source)
            variables = meta.find_undeclared_variables(parsed_content)
            template = env.get_template(DymTemplate)

            if variables:
                if self.demos is not None:
                    Uprompt = template.render(demos=self.demos, query=self.query)
                else:
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
        self.query = llmExtractor.query

    def parse(self):
        response_content = extract_json_from_response(
            self.llm_response.choices[0].message.content
        )

        # Safety check: ensure response_content is valid and has triplets
        if not response_content or not isinstance(response_content, dict):
            response_content = {"triplets": []}
        
        if "triplets" not in response_content:
            response_content["triplets"] = []

        self.output = {
            "CTI": self.query,
            "IE": response_content,
            "usage": UsageCalculator(self.config, self.llm_response).calculate(),
            "prompt": self.prompt,
            "triples_count": len(response_content["triplets"]),
        }

        return self.output


class UsageCalculator:
    def __init__(self, config, response) -> None:
        self.config = config
        self.response = response
        self.model = config.model

    def calculate(self):
        with open(resolve_path("config", "cost.json"), "r") as f:
            data = json.load(f)

        if self.model not in data:
            logger.error(
                f"Model {self.model} not found in cost.json. Setting cost to 0."
            )

        iprice = data[self.model]["input"] if self.model in data else 0
        oprice = data[self.model]["output"] if self.model in data else 0
        usageDict = {}
        usageDict["model"] = self.model

        # Handle different response formats
        if hasattr(self.response, 'usage'):
            # OpenAI format with .usage attribute
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
        elif isinstance(self.response, dict) and 'usage' in self.response:
            # Dictionary format with usage key
            usage = self.response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            usageDict["input"] = {
                "tokens": prompt_tokens,
                "cost": iprice * prompt_tokens,
            }
            usageDict["output"] = {
                "tokens": completion_tokens,
                "cost": oprice * completion_tokens,
            }
            usageDict["total"] = {
                "tokens": prompt_tokens + completion_tokens,
                "cost": iprice * prompt_tokens + oprice * completion_tokens,
            }
        else:
            # Fallback for unknown formats or missing usage info
            logger.warning("Unknown response format for usage calculation, setting tokens to 0")
            usageDict["input"] = {"tokens": 0, "cost": 0}
            usageDict["output"] = {"tokens": 0, "cost": 0}
            usageDict["total"] = {"tokens": 0, "cost": 0}

        return usageDict


class DemoRetriever:
    """
    This class is used to retrieve prompt examples for the LLMExtractor.
    """

    def __init__(self, LLMExtractor) -> None:
        self.config = LLMExtractor.config

    def retrieveRandomDemo(self, k):
        documents = []

        demo_path_parts = self.config.demoSet.split("/")
        demo_path = resolve_path(*demo_path_parts)
        
        for CTI_folder in os.listdir(demo_path):
            CTIfolderPath = os.path.join(demo_path, CTI_folder)

            for JSONfile in os.listdir(CTIfolderPath):
                with open(os.path.join(CTIfolderPath, JSONfile), "r") as f:
                    js = json.load(f)
                documents.append(
                    (
                        (
                            (js["CTI"]["text"], js["IE"]["triplets"]),
                            (JSONfile, "random"),
                        )
                    )
                )

        random.shuffle(documents)
        top_k = documents[:k]

        return [(demo[0][0], demo[0][1]) for demo in top_k], [
            (demo[1][0], demo[1][1]) for demo in top_k
        ]

    def retrievekNNDemo(self, permutation, k):
        def most_similar(doc_id, similarity_matrix):
            docs = []
            similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]

            for ix in similar_ix:
                if ix == doc_id:
                    continue

                for doc in documents:
                    if doc[0] == documents_df.iloc[ix]["documents"]:
                        docs.append((doc, similarity_matrix[doc_id][ix]))

            return docs

        documents = []

        demo_path_parts = self.config.demoSet.split("/")
        demo_path = resolve_path(*demo_path_parts)

        for JSONfile in os.listdir(demo_path):
            with open(os.path.join(demo_path, JSONfile), "r") as f:
                js = json.load(f)
                documents.append((js["text"], JSONfile))

        documents_df = pd.DataFrame(
            [doc[0] for doc in documents], columns=["documents"]
        )
        stop_words_l = stopwords.words("english")
        documents_df["documents_cleaned"] = documents_df.documents.apply(
            lambda x: " ".join(
                re.sub(r"[^a-zA-Z]", " ", w).lower()
                for w in x.split()
                if re.sub(r"[^a-zA-Z]", " ", w).lower() not in stop_words_l
            )
        )
        tfidfvectoriser = TfidfVectorizer()
        tfidfvectoriser.fit(documents_df.documents_cleaned)
        tfidf_vectors = tfidfvectoriser.transform(documents_df.documents_cleaned)
        pairwise_similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
        top_k = most_similar(0, pairwise_similarities)[:k]

        if permutation == "desc":
            return top_k

        elif permutation == "asc":
            return top_k[::-1]

    def retriveDemo(self):
        if self.config.retriever["type"] == "kNN":
            demos = self.retrievekNNDemo(
                self.config.retriever["permutation"], self.config.shot
            )
            ConsturctedDemos = []

            for demo in demos:
                demoFileName = demo[0][1]
                demoSimilarity = demo[1]

                demo_path_parts = self.config.demoSet.split("/")
                demo_path = resolve_path(*demo_path_parts)

                for JSONfile in os.listdir(demo_path):
                    if JSONfile == demoFileName:
                        with open(
                            os.path.join(demo_path, JSONfile), "r"
                        ) as f:
                            js = json.load(f)
                            ConsturctedDemos.append(
                                (
                                    (js["text"], js["explicit_triplets"]),
                                    (demoFileName, demoSimilarity),
                                )
                            )

            return [(demo[0][0], demo[0][1]) for demo in ConsturctedDemos], [
                (demo[1][0], demo[1][1]) for demo in ConsturctedDemos
            ]

        elif self.config.retriever["type"] == "rand":
            return self.retrieveRandomDemo(self.config.shot)

        else:
            logger.error('Invalid retriever type. Please choose between "kNN", "random", and "fixed".')


def extract_json_from_response(response_text):
    if isinstance(response_text, str):
        cleaned_text = response_text.strip()
        
        try:
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, TypeError):
            pass
        
        json_matches = list(
            re.finditer(r"\{[\s\S]*\}", cleaned_text.replace("\n", " "))
        )
        
        if json_matches:
            try:
                json_text = json_matches[-1].group()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    # Try to fix single quotes to double quotes
                    fixed_json = json_text.replace("'", '"')
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        # Remove any trailing commas and fix common issues
                        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                        fixed_json = re.sub(r'([{,]\s*)(\w+)(\s*):', r'\1"\2"\3:', fixed_json)
                        return json.loads(fixed_json)
                        
            except Exception as e:
                logger.error(f"Error extracting JSON from match: {e}")
                logger.debug(f"JSON text: {json_matches[-1].group()}")
        
        # Try to parse as triplets list format
        triplet_patterns = [
            r"\{'subject':\s*'([^']*)',\s*'relation':\s*'([^']*)',\s*'object':\s*'([^']*)'\}",
            r'\{"subject":\s*"([^"]*)",\s*"relation":\s*"([^"]*)",\s*"object":\s*"([^"]*)"\}',
            r"'subject':\s*'([^']*)',\s*'relation':\s*'([^']*)',\s*'object':\s*'([^']*)'",
            r'"subject":\s*"([^"]*)",\s*"relation":\s*"([^"]*)",\s*"object":\s*"([^"]*)"'
        ]
        
        for pattern in triplet_patterns:
            triplet_matches = re.findall(pattern, cleaned_text)
            if triplet_matches:
                # Convert to expected format
                triplets = []
                for match in triplet_matches:
                    subject, relation, obj = match
                    triplets.append({
                        "subject": subject.strip(),
                        "relation": relation.strip(),
                        "object": obj.strip()
                    })
                return {"triplets": triplets}
        
        logger.warning(f"Failed to parse response, raw text: {response_text}")
        return {"triplets": []}
    else:
        return dict(response_text)
