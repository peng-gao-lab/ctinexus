import json
import logging
import os
import random
import re
import time
from functools import wraps
from html import unescape
from urllib.parse import urlparse

import litellm
import nltk
import numpy as np
import pandas as pd
import trafilatura
from jinja2 import Environment, FileSystemLoader, meta
from nltk.corpus import stopwords
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from ctinexus.utils.path_utils import resolve_path

logger = logging.getLogger(__name__)

_STOPWORDS_CACHE = None

litellm.drop_params = True
_CUSTOM_ENDPOINT_LOGGED = False


def get_litellm_endpoint_overrides(default_api_base: str = None) -> dict:
	"""Resolve LiteLLM endpoint overrides from environment variables."""
	global _CUSTOM_ENDPOINT_LOGGED

	custom_base_url = (os.getenv("CUSTOM_BASE_URL") or "").strip()
	if custom_base_url:
		custom_api_key = os.getenv("CUSTOM_API_KEY")
		if not _CUSTOM_ENDPOINT_LOGGED:
			logger.info("Using custom LLM endpoint: %s", custom_base_url)
			_CUSTOM_ENDPOINT_LOGGED = True
		return {
			"api_base": custom_base_url,
			"api_key": custom_api_key if custom_api_key is not None else "",
		}

	if default_api_base:
		return {"api_base": default_api_base}

	return {}


def call_litellm_completion(model: str, *, default_api_base: str = None, **completion_kwargs):
	"""Call LiteLLM completion with optional endpoint overrides."""
	request_kwargs = dict(completion_kwargs)
	request_kwargs.update(get_litellm_endpoint_overrides(default_api_base))
	return litellm.completion(model=model, **request_kwargs)


def get_english_stopwords():
	"""Load English stopwords lazily to avoid import-time side effects."""
	global _STOPWORDS_CACHE
	if _STOPWORDS_CACHE is not None:
		return _STOPWORDS_CACHE

	try:
		_STOPWORDS_CACHE = set(stopwords.words("english"))
		return _STOPWORDS_CACHE
	except LookupError:
		logger.info("NLTK stopwords corpus not found. Downloading...")

	try:
		nltk.download("stopwords", quiet=True)
		_STOPWORDS_CACHE = set(stopwords.words("english"))
	except Exception as e:
		logger.warning(f"Unable to load NLTK stopwords. Continuing without stopword filtering: {e}")
		_STOPWORDS_CACHE = set()

	return _STOPWORDS_CACHE


def validate_triplet(triplet: dict) -> bool:
	"""Validate that a triplet has the required structure.

	A valid triplet must have 'subject', 'relation', and 'object' keys.
	For IE stage: subject/object should be strings or have 'text' key.
	For ET stage: subject/object should have 'text' and 'class' keys.
	For EA stage: subject/object should have 'mention_text', 'mention_id', etc.
	"""
	if not isinstance(triplet, dict):
		return False

	required_keys = ["subject", "relation", "object"]
	if not all(key in triplet for key in required_keys):
		return False

	# Check subject and object are valid (either strings or dicts with required keys)
	for key in ["subject", "object"]:
		value = triplet[key]
		if value is None:
			return False
		if isinstance(value, str):
			if not value.strip():
				return False
		elif isinstance(value, dict):
			# Must have at least 'text' or 'mention_text'
			if not (value.get("text") or value.get("mention_text")):
				return False
		else:
			return False

	# Relation should be a non-empty string
	relation = triplet.get("relation")
	if not isinstance(relation, str) or not relation.strip():
		return False

	return True


def filter_valid_triplets(triplets: list, stage: str = "unknown") -> list:
	"""Filter out invalid triplets and log warnings for dropped ones."""
	if not triplets:
		return []

	valid_triplets = []
	for i, triplet in enumerate(triplets):
		if validate_triplet(triplet):
			valid_triplets.append(triplet)
		else:
			logger.warning(f"[{stage}] Dropping invalid triplet at index {i}: {triplet}")

	if len(valid_triplets) < len(triplets):
		logger.warning(
			f"[{stage}] Filtered {len(triplets) - len(valid_triplets)} invalid triplets, "
			f"{len(valid_triplets)} remaining"
		)

	return valid_triplets


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
						logger.debug("Retrying...")
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
		self.response_content = extract_json_from_response(self.response.choices[0].message.content)

		# Safety check: ensure response_content has required keys
		if not self.response_content or not isinstance(self.response_content, dict):
			self.response_content = {"tagged_triples": []}

		if "tagged_triples" not in self.response_content:
			# Try alternative key names that models might use
			if "triplets" in self.response_content:
				self.response_content["tagged_triples"] = self.response_content["triplets"]
			else:
				self.response_content["tagged_triples"] = []

		# Validate and filter triplets
		tagged_triplets = self.response_content.get("tagged_triples", [])
		if not isinstance(tagged_triplets, list):
			logger.warning("[ET] tagged_triples is not a list, resetting to empty")
			tagged_triplets = []
		tagged_triplets = filter_valid_triplets(tagged_triplets, stage="ET")

		result["ET"] = {}
		result["ET"]["typed_triplets"] = tagged_triplets
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


class UrlSourceInput:
	def __init__(self, config: DictConfig):
		self.config = config

	def call(self, source_url: str) -> dict:
		if not isinstance(source_url, str) or not source_url.strip():
			return self._build_error("invalid_url", "URL input is empty or not a string.", source_url)

		normalized_url = self._normalize_url(source_url)
		if not self._is_valid_url(normalized_url):
			return self._build_error("invalid_url", "URL must be a valid http/https address.", normalized_url)

		try:
			html_content = trafilatura.fetch_url(normalized_url)
		except Exception as e:
			logger.error(f"Failed to fetch URL {normalized_url}: {e}")
			return self._build_error("fetch_failed", f"Failed to fetch URL content: {e}", normalized_url)

		if not html_content:
			return self._build_error("fetch_failed", "No content returned while fetching URL.", normalized_url)

		hybrid_extract = self.extract_hybrid_content(html_content)
		if not hybrid_extract.get("text"):
			return self._build_error(
				"extraction_failed", "No readable article/report content could be extracted.", normalized_url
			)

		raw_text = hybrid_extract.get("text", "")
		normalized_text = self.normalize_text(raw_text)
		if not normalized_text:
			return self._build_error(
				"empty_content",
				"Extracted content was empty after normalization.",
				normalized_url,
			)
		focused_text = self.build_cti_focus_text(normalized_text)

		metadata = {
			"title": hybrid_extract.get("title"),
			"author": hybrid_extract.get("author"),
			"date": hybrid_extract.get("date"),
			"source_domain": self._extract_domain(normalized_url),
			"url": normalized_url,
		}

		result = {
			"status": "success",
			"source_type": "url",
			"url": normalized_url,
			"source_domain": metadata["source_domain"],
			"metadata": metadata,
			"raw_text_length": len(raw_text.strip()),
			"prompt_template": getattr(self.config, "url_prompt_file", None) or "url_source_input.jinja",
		}

		summary_prompt = self.generate_prompt(
			result,
			raw_text=raw_text,
			html_content=html_content,
			normalized_text=normalized_text,
			focused_text=focused_text,
		)

		summary_generated = False
		summary_usage = None
		try:
			summary_text, summary_response_time, summary_usage = self.summarize(summary_prompt)
			result["summarized_text"] = self.normalize_summary_text(summary_text)
			result["summary_response_time"] = summary_response_time
			result["summary_model_usage"] = summary_usage
			summary_generated = True
		except Exception as e:
			logger.warning(f"URL summarization failed, falling back to normalized text: {e}")
			result["summarized_text"] = normalized_text
			result["summary_response_time"] = 0
			result["summary_model_usage"] = None

		# Force paragraph-only CTI output if first pass drifts to bullets/headers.
		if summary_generated and not self.is_well_formed_cti_paragraph(result["summarized_text"]):
			try:
				repaired_text, repaired_time, repaired_usage = self.repair_summary(
					source_result=result,
					initial_summary=result["summarized_text"],
					focused_text=focused_text,
				)
				repaired_summary = self.normalize_summary_text(repaired_text)
				if repaired_summary:
					result["summarized_text"] = repaired_summary
					result["summary_response_time"] += repaired_time
					result["summary_model_usage"] = self.merge_usages(summary_usage, repaired_usage)
				else:
					logger.warning("URL summary repair returned empty text; retaining initial summary.")
			except Exception as e:
				logger.warning(f"URL summary repair failed; retaining initial summary: {e}")

		result["final_text"] = result["summarized_text"] or normalized_text

		return result

	def generate_prompt(
		self,
		source_result: dict,
		raw_text: str = "",
		html_content: str = "",
		normalized_text: str = "",
		focused_text: str = "",
	) -> list[dict]:
		"""Generate prompt payload for downstream summarization stages."""
		url_prompt_folder = getattr(self.config, "url_prompt_folder", None) or "prompts"
		url_prompt_file = getattr(self.config, "url_prompt_file", None) or "url_source_input.jinja"

		try:
			env = Environment(loader=FileSystemLoader(resolve_path(url_prompt_folder)))
			template_file = env.loader.get_source(env, url_prompt_file)[0]
			template = env.get_template(url_prompt_file)
			variables = meta.find_undeclared_variables(env.parse(template_file))

			prompt_context = {
				"url": source_result.get("url"),
				"source_url": source_result.get("url"),
				"source_domain": source_result.get("source_domain"),
				"metadata": source_result.get("metadata", {}),
				"title": source_result.get("metadata", {}).get("title"),
				"author": source_result.get("metadata", {}).get("author"),
				"date": source_result.get("metadata", {}).get("date"),
				"published_date": source_result.get("metadata", {}).get("date"),
				"content": focused_text or normalized_text,
				"normalized_text": normalized_text,
				"focused_text": focused_text or normalized_text,
				"raw_text": raw_text,
				"html_content": html_content,
				"raw_text_length": source_result.get("raw_text_length", 0),
				"normalized_text_length": len(normalized_text),
				"focused_text_length": len(focused_text),
			}

			if variables:
				user_prompt = template.render(**prompt_context)
			else:
				user_prompt = template.render()
			return [{"role": "user", "content": user_prompt}]
		except Exception as e:
			logger.warning(f"URL prompt generation failed, falling back to plain content prompt: {e}")
			return [{"role": "user", "content": focused_text or normalized_text}]

	def extract_hybrid_content(self, html_content: str) -> dict:
		"""Hybrid extraction: trafilatura variants + metadata/script fallbacks + CTI-aware merge."""
		candidate_bodies = []
		metadata = {"title": None, "author": None, "date": None}

		json_extract = self._extract_trafilatura_json(html_content)
		if json_extract:
			self._merge_metadata(metadata, json_extract)
			text = json_extract.get("text")
			if text:
				candidate_bodies.append({"source": "trafilatura_json", "text": text})

		txt_extract = self._extract_trafilatura_text(html_content)
		if txt_extract:
			candidate_bodies.append({"source": "trafilatura_txt", "text": txt_extract})

		bare_extract = self._extract_trafilatura_bare(html_content)
		if bare_extract:
			self._merge_metadata(metadata, bare_extract)
			text = bare_extract.get("text")
			if text:
				candidate_bodies.append({"source": "trafilatura_bare", "text": text})

		jsonld_extract = self._extract_jsonld_text(html_content)
		if jsonld_extract:
			self._merge_metadata(metadata, jsonld_extract)
			text = jsonld_extract.get("text")
			if text:
				candidate_bodies.append({"source": "jsonld", "text": text})

		meta_extract = self._extract_meta_description(html_content)
		if meta_extract:
			text = meta_extract.get("text")
			if text:
				candidate_bodies.append({"source": "meta_description", "text": text})

		normalized_candidates = []
		for item in candidate_bodies:
			normalized = self.normalize_text(item["text"])
			if not normalized:
				continue
			normalized_candidates.append(
				{
					"source": item["source"],
					"text": normalized,
					"length": len(normalized),
				}
			)

		merged_text = self.merge_extraction_candidates(normalized_candidates)
		return {
			"text": merged_text,
			"title": metadata.get("title"),
			"author": metadata.get("author"),
			"date": metadata.get("date"),
			"candidates": [{k: v for k, v in c.items() if k != "text"} for c in normalized_candidates],
		}

	def merge_extraction_candidates(self, candidates: list, max_chars: int = 16000) -> str:
		if not candidates:
			return ""

		source_priority = {
			"trafilatura_json": 0,
			"trafilatura_bare": 1,
			"trafilatura_txt": 2,
			"readability": 3,
			"jsonld": 4,
			"meta_description": 5,
		}
		candidates_sorted = sorted(
			candidates,
			key=lambda x: (source_priority.get(x["source"], 99), -x["length"]),
		)
		base_text = candidates_sorted[0]["text"]
		base_lines = [line.strip() for line in base_text.splitlines() if line.strip()]
		seen = {line.lower() for line in base_lines}
		merged_lines = list(base_lines)

		for candidate in candidates_sorted[1:]:
			for line in candidate["text"].splitlines():
				line = line.strip()
				if not line:
					continue
				line_key = line.lower()
				if line_key in seen:
					continue
				if not self.is_cti_signal_line(line):
					continue
				seen.add(line_key)
				merged_lines.append(line)

		merged = "\n".join(merged_lines).strip()
		if len(merged) > max_chars:
			merged = merged[:max_chars].rsplit(" ", 1)[0].strip()
		return merged

	def is_cti_signal_line(self, line: str) -> bool:
		lower_line = line.lower()
		drop_patterns = [
			r"^figure\s+\d+",
			r"^source:",
			r"^sources:",
			r"^cookie",
			r"^subscribe",
			r"^sign up",
			r"^read more",
			r"^trend micro solutions?",
			r"^here are some security best practices",
			r"^recommendations?$",
		]
		if any(re.match(pattern, lower_line) for pattern in drop_patterns):
			return False

		strong_terms = [
			"ransomware",
			"threat actor",
			"campaign",
			"extortion",
			"raas",
			"cve-",
			"exploit",
			"vulnerability",
			"cobalt strike",
			"mimikatz",
			"psexec",
			"anydesk",
			"rclone",
			"winscp",
			"linux",
			"esxi",
			"vpn",
			"initial access",
			"persistence",
			"defense evasion",
			"lateral movement",
			"command and control",
			"exfiltration",
			"impact",
			"encrypt",
			"leak site",
			"conti",
			"ryuk",
			"akira",
			"victim",
			"compromis",
			"credential",
			"double extortion",
			"tor",
			"hc3",
			"cisco",
		]
		if any(term in lower_line for term in strong_terms):
			return True

		# Preserve descriptive narrative lines that are sentence-like and non-trivial.
		if len(line) >= 90 and re.search(r"[.!?]$", line):
			return True
		return False

	def _extract_trafilatura_json(self, html_content: str) -> dict:
		try:
			result = trafilatura.extract(
				html_content,
				output_format="json",
				with_metadata=True,
				include_comments=False,
				include_tables=False,
				deduplicate=True,
				favor_recall=True,
			)
			if not result:
				return {}
			if isinstance(result, str):
				try:
					return json.loads(result)
				except json.JSONDecodeError:
					return {"text": result}
			return result
		except Exception as e:
			logger.debug(f"Trafilatura JSON extraction failed: {e}")
			return {}

	def _extract_trafilatura_text(self, html_content: str) -> str:
		try:
			result = trafilatura.extract(
				html_content,
				output_format="txt",
				include_comments=False,
				include_tables=False,
				deduplicate=True,
				favor_recall=True,
			)
			return result or ""
		except Exception as e:
			logger.debug(f"Trafilatura text extraction failed: {e}")
			return ""

	def _extract_trafilatura_bare(self, html_content: str) -> dict:
		if not hasattr(trafilatura, "bare_extraction"):
			return {}

		try:
			result = trafilatura.bare_extraction(
				html_content,
				with_metadata=True,
				include_comments=False,
				include_tables=False,
				deduplicate=True,
				favor_recall=True,
			)
			if isinstance(result, dict):
				return result
			return {}
		except Exception as e:
			logger.debug(f"Trafilatura bare extraction failed: {e}")
			return {}

	def _extract_jsonld_text(self, html_content: str) -> dict:
		jsonld_scripts = re.findall(
			r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
			html_content,
			flags=re.IGNORECASE | re.DOTALL,
		)
		collected_text = []
		collected_title = None
		collected_date = None
		collected_author = None

		for script_text in jsonld_scripts:
			payload = script_text.strip()
			if not payload:
				continue

			try:
				parsed = json.loads(payload)
			except json.JSONDecodeError:
				# Retry for occasional malformed but recoverable script payloads.
				cleaned = re.sub(r"[\x00-\x1f]", "", payload)
				try:
					parsed = json.loads(cleaned)
				except json.JSONDecodeError:
					continue

			for item in self._iterate_jsonld_nodes(parsed):
				article_body = item.get("articleBody")
				description = item.get("description")
				headline = item.get("headline")
				date_published = item.get("datePublished")
				author = item.get("author")

				if article_body and isinstance(article_body, str):
					collected_text.append(article_body)
				if description and isinstance(description, str):
					collected_text.append(description)
				if not collected_title and isinstance(headline, str):
					collected_title = headline
				if not collected_date and isinstance(date_published, str):
					collected_date = date_published
				if not collected_author:
					collected_author = self._normalize_jsonld_author(author)

		return {
			"title": collected_title,
			"date": collected_date,
			"author": collected_author,
			"text": "\n".join(collected_text).strip(),
		}

	def _extract_meta_description(self, html_content: str) -> dict:
		patterns = [
			r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
			r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
		]
		for pattern in patterns:
			match = re.search(pattern, html_content, flags=re.IGNORECASE)
			if match:
				return {"text": unescape(match.group(1).strip())}
		return {}

	def _merge_metadata(self, target: dict, source: dict):
		for key in ["title", "author", "date"]:
			if not target.get(key) and source.get(key):
				target[key] = source.get(key)

	def _iterate_jsonld_nodes(self, value):
		if isinstance(value, dict):
			yield value
			graph = value.get("@graph")
			if isinstance(graph, list):
				for item in graph:
					yield from self._iterate_jsonld_nodes(item)
		elif isinstance(value, list):
			for item in value:
				yield from self._iterate_jsonld_nodes(item)

	def _normalize_jsonld_author(self, author_value):
		if isinstance(author_value, str):
			return author_value
		if isinstance(author_value, dict):
			return author_value.get("name")
		if isinstance(author_value, list):
			names = []
			for item in author_value:
				if isinstance(item, str):
					names.append(item)
				elif isinstance(item, dict) and item.get("name"):
					names.append(item.get("name"))
			return ", ".join(names) if names else None
		return None

	def summarize(self, prompt: list[dict]) -> tuple[str, float, dict]:
		"""Summarize URL content into CTINexus-ready text using configured model."""
		start_time = time.time()
		provider = self.config.provider.lower()
		model_id = self.config.model

		completion_kwargs = {
			"messages": prompt,
			"temperature": 0.0,
		}

		if provider == "gemini":
			response = call_litellm_completion(model=f"gemini/{model_id}", **completion_kwargs)
		elif provider == "ollama":
			ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
			response = call_litellm_completion(
				model=f"ollama/{model_id}",
				default_api_base=ollama_base_url,
				**completion_kwargs,
			)
		else:
			response = call_litellm_completion(model=model_id, **completion_kwargs)

		response_time = time.time() - start_time
		summary_text = response.choices[0].message.content.strip() if response.choices else ""
		if not summary_text:
			raise ValueError("Empty summary response from URL summarization model")
		usage = UsageCalculator(self.config, response).calculate()
		return summary_text, response_time, usage

	def repair_summary(self, source_result: dict, initial_summary: str, focused_text: str) -> tuple[str, float, dict]:
		"""Repair non-compliant summary output into strict CTINexus-ready paragraph format."""
		repair_prompt = (
			"You will rewrite a CTI summary into strict format.\n"
			"Rules:\n"
			"1) Return exactly one paragraph with 3 to 6 sentences.\n"
			"2) No markdown, no bullets, no headings, no preface.\n"
			"3) Keep only verifiable CTI facts from the source.\n"
			"4) Preserve concrete identifiers if present (CVE IDs, malware/tool names, actor/campaign names).\n"
			"5) Remove vendor marketing, generic recommendations, and product promotion.\n\n"
			f"Source URL: {source_result.get('url')}\n"
			f"Source Domain: {source_result.get('source_domain')}\n"
			f"Title: {source_result.get('metadata', {}).get('title')}\n\n"
			"Original summary to fix:\n"
			f"{initial_summary}\n\n"
			"Relevant extracted CTI content:\n"
			f"{focused_text[:9000]}\n\n"
			"Output only the rewritten paragraph."
		)
		return self.summarize([{"role": "user", "content": repair_prompt}])

	def build_cti_focus_text(self, normalized_text: str, max_chars: int = 10000) -> str:
		"""Trim obvious non-CTI sections while preserving attack chain details."""
		lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
		skip_patterns = [
			r"^recommendations?$",
			r"^trend micro solutions?$",
			r"^to protect systems against similar threats",
			r"^here are some best practices",
			r"^audit and inventory$",
			r"^configure and monitor$",
			r"^patch and update$",
			r"^protect and recover$",
			r"^secure and defend$",
			r"^about trend micro$",
			r"^copyright",
		]
		drop_line_patterns = [
			r"^figure\s+\d+",
			r"^source:",
			r"^sources:",
			r"^read more",
			r"^related",
			r"^subscribe",
		]

		filtered = []
		skip_section = False
		for line in lines:
			line_lower = line.lower()
			if any(re.match(pat, line_lower) for pat in skip_patterns):
				skip_section = True
				continue
			if skip_section:
				# Resume when we hit core technical section headers
				if re.match(
					r"^(infection chain and techniques|initial access|execution|defense evasion|lateral movement|command and control|exfiltration|impact|other technical details)",
					line_lower,
				):
					skip_section = False
				else:
					continue
			if any(re.match(pat, line_lower) for pat in drop_line_patterns):
				continue
			filtered.append(line)

		focused = "\n".join(filtered).strip()
		if len(focused) > max_chars:
			focused = focused[:max_chars].rsplit(" ", 1)[0].strip()
		return focused

	def normalize_summary_text(self, summary_text: str) -> str:
		"""Normalize summary output to a single clean paragraph string."""
		if not isinstance(summary_text, str):
			return ""
		s = summary_text.strip()
		s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
		s = re.sub(r"\s*```$", "", s)
		s = re.sub(r"\s+", " ", s).strip()
		return s

	def is_well_formed_cti_paragraph(self, summary_text: str) -> bool:
		"""Check strict formatting/quality expectations for CTINexus-ready summary text."""
		if not summary_text or not isinstance(summary_text, str):
			return False

		text = summary_text.strip()
		if len(text) < 120:
			return False
		if re.search(r"(^|\n)\s*[-*+]\s+", text):
			return False
		if re.search(r"(^|\n)\s*#{1,6}\s+", text):
			return False
		if text.lower().startswith(("here is", "here's", "summary:", "in summary")):
			return False

		sentences = re.split(r"(?<=[.!?])\s+", text)
		sentence_count = len([s for s in sentences if s.strip()])
		if sentence_count < 3 or sentence_count > 6:
			return False
		return True

	def merge_usages(self, usage_a: dict, usage_b: dict) -> dict:
		if not usage_a:
			return usage_b
		if not usage_b:
			return usage_a
		if usage_a.get("model") != usage_b.get("model"):
			return usage_b

		return {
			"model": usage_a["model"],
			"input": {
				"tokens": usage_a["input"]["tokens"] + usage_b["input"]["tokens"],
				"cost": usage_a["input"]["cost"] + usage_b["input"]["cost"],
			},
			"output": {
				"tokens": usage_a["output"]["tokens"] + usage_b["output"]["tokens"],
				"cost": usage_a["output"]["cost"] + usage_b["output"]["cost"],
			},
			"total": {
				"tokens": usage_a["total"]["tokens"] + usage_b["total"]["tokens"],
				"cost": usage_a["total"]["cost"] + usage_b["total"]["cost"],
			},
		}

	def normalize_text(self, extracted_text: str) -> str:
		"""Light cleanup to reduce web boilerplate and normalize whitespace."""
		if not isinstance(extracted_text, str):
			return ""

		cleaned = unescape(extracted_text).replace("\r\n", "\n").replace("\r", "\n")
		cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", cleaned)

		boilerplate_patterns = [
			r"^\s*cookie(s)?\b",
			r"^\s*accept (all )?cookies\b",
			r"^\s*privacy policy\b",
			r"^\s*terms (of use|and conditions)\b",
			r"^\s*subscribe\b",
			r"^\s*sign up\b",
			r"^\s*advertisement\b",
			r"^\s*all rights reserved\b",
		]

		normalized_lines = []
		seen = set()
		for line in cleaned.splitlines():
			line = re.sub(r"\s+", " ", line).strip()
			if not line:
				continue
			if any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in boilerplate_patterns):
				continue
			line_key = line.lower()
			if line_key in seen:
				continue
			seen.add(line_key)
			normalized_lines.append(line)

		normalized = "\n".join(normalized_lines)
		normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
		return normalized

	def _normalize_url(self, source_url: str) -> str:
		url = source_url.strip()
		parsed = urlparse(url)
		if not parsed.scheme:
			url = f"https://{url}"
		return url

	def _is_valid_url(self, source_url: str) -> bool:
		parsed = urlparse(source_url)
		return parsed.scheme in {"http", "https"} and bool(parsed.netloc and " " not in parsed.netloc)

	def _extract_domain(self, source_url: str) -> str:
		return urlparse(source_url).netloc.lower()

	def _build_error(self, code: str, message: str, source_url: str = None) -> dict:
		return {
			"status": "error",
			"source_type": "url",
			"url": source_url,
			"source_domain": self._extract_domain(source_url)
			if source_url and self._is_valid_url(source_url)
			else None,
			"metadata": {
				"title": None,
				"author": None,
				"date": None,
				"source_domain": None,
				"url": source_url,
			},
			"raw_text_length": 0,
			"prompt_template": getattr(self.config, "url_prompt_file", None) or "url_source_input.jinja",
			"error": {"code": code, "message": message},
		}


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
			self.response_content = extract_json_from_response(self.llm_response.choices[0].message.content)

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

			if pred_sub == main_node["entity_text"] and pred_obj == self.topic_node["entity_text"]:
				new_sub = {
					"entity_id": main_node["entity_id"],
					"mention_text": main_node["entity_text"],
				}
				new_obj = self.topic_node
			elif pred_obj == main_node["entity_text"] and pred_sub == self.topic_node["entity_text"]:
				new_sub = self.topic_node
				new_obj = {
					"entity_id": main_node["entity_id"],
					"mention_text": main_node["entity_text"],
				}
			else:
				logger.debug(
					"The predicted subject and object do not match the unvisited subject and topic entity, the LLM produce hallucination!"
				)
				logger.debug(f"Hallucinated in text: {self.js['text']}")

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
		parsed_template = env.parse(env.loader.get_source(env, self.config.link_prompt_file)[0])
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
				response = call_litellm_completion(
					model=model_id,
					messages=messages,
					max_tokens=self.max_tokens,
					response_format={"type": "json_object"},
				)
			elif provider == "gemini":
				response = call_litellm_completion(
					model=f"gemini/{model_id}",
					messages=[{"role": "user", "content": self.prompt[-1]["content"]}],
					max_tokens=self.max_tokens,
					temperature=0.8,
					response_format={"type": "json_object"},
				)
			elif provider == "meta":
				response = call_litellm_completion(
					model=model_id,
					messages=[{"role": "user", "content": self.prompt[-1]["content"]}],
					max_tokens=self.max_tokens,
					temperature=0.8,
					top_p=0.9,
				)
			elif provider == "ollama":
				ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

				improved_prompt = (
					self.prompt[-1]["content"]
					+ "\n\nIMPORTANT: output should be a valid JSON object with no extra text or description."
				)

				response = call_litellm_completion(
					model=f"ollama/{model_id}",
					messages=[{"role": "user", "content": improved_prompt}],
					max_tokens=self.max_tokens,
					temperature=0.8,
					default_api_base=ollama_base_url,
				)
			else:
				response = call_litellm_completion(
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
		self.llm_response, self.response_time = LLMCaller(self.config, self.prompt).call()

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
				outJSON["IE"]["Prompt"]["permutation"] = self.config.retriever.permutation
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
				raise ValueError(f"Invalid template directory: {self.config.ie_prompt_set}")

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
		response_content = extract_json_from_response(self.llm_response.choices[0].message.content)

		# Safety check: ensure response_content is valid and has triplets
		if not response_content or not isinstance(response_content, dict):
			response_content = {"triplets": []}

		if "triplets" not in response_content:
			response_content["triplets"] = []

		# Validate and filter triplets from IE stage
		triplets = response_content.get("triplets", [])
		if not isinstance(triplets, list):
			logger.warning("[IE] triplets is not a list, resetting to empty")
			triplets = []
		triplets = filter_valid_triplets(triplets, stage="IE")
		response_content["triplets"] = triplets

		self.output = {
			"CTI": self.query,
			"IE": response_content,
			"usage": UsageCalculator(self.config, self.llm_response).calculate(),
			"prompt": self.prompt,
			"triples_count": len(triplets),
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
			logger.warning(f"Model {self.model} not found in cost.json. Setting cost to 0.")

		iprice = data[self.model]["input"] if self.model in data else 0
		oprice = data[self.model]["output"] if self.model in data else 0
		usageDict = {}
		usageDict["model"] = self.model

		# Handle different response formats
		if hasattr(self.response, "usage"):
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
				"tokens": self.response.usage.prompt_tokens + self.response.usage.completion_tokens,
				"cost": iprice * self.response.usage.prompt_tokens + oprice * self.response.usage.completion_tokens,
			}
		elif isinstance(self.response, dict) and "usage" in self.response:
			# Dictionary format with usage key
			usage = self.response["usage"]
			prompt_tokens = usage.get("prompt_tokens", 0)
			completion_tokens = usage.get("completion_tokens", 0)

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

		return [(demo[0][0], demo[0][1]) for demo in top_k], [(demo[1][0], demo[1][1]) for demo in top_k]

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

		documents_df = pd.DataFrame([doc[0] for doc in documents], columns=["documents"])
		stop_words_l = get_english_stopwords()
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
			demos = self.retrievekNNDemo(self.config.retriever["permutation"], self.config.shot)
			ConsturctedDemos = []

			for demo in demos:
				demoFileName = demo[0][1]
				demoSimilarity = demo[1]

				demo_path_parts = self.config.demoSet.split("/")
				demo_path = resolve_path(*demo_path_parts)

				for JSONfile in os.listdir(demo_path):
					if JSONfile == demoFileName:
						with open(os.path.join(demo_path, JSONfile), "r") as f:
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

		json_matches = list(re.finditer(r"\{[\s\S]*\}", cleaned_text.replace("\n", " ")))

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
						fixed_json = re.sub(r",(\s*[}\]])", r"\1", fixed_json)
						fixed_json = re.sub(r"([{,]\s*)(\w+)(\s*):", r'\1"\2"\3:', fixed_json)
						return json.loads(fixed_json)

			except Exception as e:
				logger.error(f"Error extracting JSON from match: {e}")
				logger.debug(f"JSON text: {json_matches[-1].group()}")

		# Try to parse as triplets list format
		triplet_patterns = [
			r"\{'subject':\s*'([^']*)',\s*'relation':\s*'([^']*)',\s*'object':\s*'([^']*)'\}",
			r'\{"subject":\s*"([^"]*)",\s*"relation":\s*"([^"]*)",\s*"object":\s*"([^"]*)"\}',
			r"'subject':\s*'([^']*)',\s*'relation':\s*'([^']*)',\s*'object':\s*'([^']*)'",
			r'"subject":\s*"([^"]*)",\s*"relation":\s*"([^"]*)",\s*"object":\s*"([^"]*)"',
		]

		for pattern in triplet_patterns:
			triplet_matches = re.findall(pattern, cleaned_text)
			if triplet_matches:
				# Convert to expected format
				triplets = []
				for match in triplet_matches:
					subject, relation, obj = match
					triplets.append({"subject": subject.strip(), "relation": relation.strip(), "object": obj.strip()})
				return {"triplets": triplets}

		logger.warning(f"Failed to parse response, raw text: {response_text}")
		raise ValueError("Failed to extract JSON from response text")
	else:
		return dict(response_text)
