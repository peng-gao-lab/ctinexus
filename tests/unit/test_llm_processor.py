"""Unit tests for ctinexus.llm_processor module."""

from omegaconf import OmegaConf

from ctinexus.llm_processor import (
	UrlSourceInput,
	UsageCalculator,
	extract_json_from_response,
)


class TestExtractJsonFromResponse:
	"""Test JSON extraction from LLM responses."""

	def test_extract_valid_json_string(self):
		"""Test extracting valid JSON from string."""
		response = '{"triplets": [{"subject": "APT29", "relation": "uses", "object": "PowerShell"}]}'
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result
		assert len(result["triplets"]) == 1

	def test_extract_json_with_whitespace(self):
		"""Test extracting JSON with leading/trailing whitespace."""
		response = '   {"triplets": []}   '
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_json_from_text_with_markers(self):
		"""Test extracting JSON from text with markdown code blocks."""
		response = """Here is the result:
		```json
		{"triplets": [{"subject": "test", "relation": "rel", "object": "obj"}]}
		```
		"""
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_json_embedded_in_text(self):
		"""Test extracting JSON embedded in regular text."""
		response = 'Some text before {"triplets": []} some text after'
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_json_with_single_quotes(self):
		"""Test extracting JSON with single quotes (should be converted)."""
		response = "{'triplets': [{'subject': 'test', 'relation': 'rel', 'object': 'obj'}]}"
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_json_with_trailing_comma(self):
		"""Test extracting JSON with trailing commas."""
		response = '{"triplets": [{"subject": "test",}]}'
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_json_dict_input(self):
		"""Test that dict input is returned as-is."""
		response_dict = {"triplets": [{"subject": "test"}]}
		result = extract_json_from_response(response_dict)

		assert result == response_dict

	def test_extract_multiple_json_objects(self):
		"""Test extracting when multiple JSON objects are present (uses first valid JSON)."""
		response = '{"first": "object"}'
		result = extract_json_from_response(response)

		assert result is not None
		assert result == {"first": "object"}

	def test_extract_json_with_unquoted_keys(self):
		"""Test extracting JSON with unquoted keys."""
		response = "{triplets: [{subject: 'test', relation: 'rel', object: 'obj'}]}"
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result

	def test_extract_triplet_pattern(self):
		"""Test extracting triplets using pattern matching."""
		response = "'subject': 'APT29', 'relation': 'uses', 'object': 'PowerShell'"
		result = extract_json_from_response(response)

		assert result is not None
		assert "triplets" in result
		assert len(result["triplets"]) > 0


class TestUsageCalculator:
	"""Test usage calculation for LLM responses."""

	def test_usage_calculator_basic(self, sample_llm_response):
		"""Test basic usage calculation."""
		config = OmegaConf.create({"model": "test-model"})
		calculator = UsageCalculator(config, sample_llm_response)

		result = calculator.calculate()

		assert "model" in result
		assert result["model"] == "test-model"
		assert "input" in result
		assert "output" in result
		assert "total" in result

	def test_usage_calculator_with_tokens(self, sample_llm_response):
		"""Test that calculator correctly extracts token counts."""
		config = OmegaConf.create({"model": "gpt-4o"})
		calculator = UsageCalculator(config, sample_llm_response)

		result = calculator.calculate()

		assert result["input"]["tokens"] == 100
		assert result["output"]["tokens"] == 50
		assert result["total"]["tokens"] == 150

	def test_usage_calculator_calculates_cost(self, sample_llm_response):
		"""Test that calculator computes costs correctly."""
		config = OmegaConf.create({"model": "gpt-4o"})
		calculator = UsageCalculator(config, sample_llm_response)

		result = calculator.calculate()

		# Costs should be calculated based on token counts
		assert "cost" in result["input"]
		assert "cost" in result["output"]
		assert "cost" in result["total"]
		assert result["input"]["cost"] >= 0
		assert result["output"]["cost"] >= 0
		assert result["total"]["cost"] >= 0

	def test_usage_calculator_unknown_model(self):
		"""Test usage calculator with unknown model (should set cost to 0)."""

		class MockResponse:
			def __init__(self):
				self.usage = MockUsage()

		class MockUsage:
			def __init__(self):
				self.prompt_tokens = 100
				self.completion_tokens = 50

		config = OmegaConf.create({"model": "unknown-model"})
		calculator = UsageCalculator(config, MockResponse())

		result = calculator.calculate()

		assert result["input"]["cost"] == 0
		assert result["output"]["cost"] == 0
		assert result["total"]["cost"] == 0

	def test_usage_calculator_dict_response(self):
		"""Test usage calculator with dictionary response format."""
		response = {
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 50,
			}
		}

		config = OmegaConf.create({"model": "test-model"})
		calculator = UsageCalculator(config, response)

		result = calculator.calculate()

		assert result["input"]["tokens"] == 100
		assert result["output"]["tokens"] == 50
		assert result["total"]["tokens"] == 150

	def test_usage_calculator_missing_usage(self):
		"""Test usage calculator with missing usage information."""

		class MockResponse:
			pass

		config = OmegaConf.create({"model": "test-model"})
		calculator = UsageCalculator(config, MockResponse())

		result = calculator.calculate()

		# Should default to 0 for missing usage info
		assert result["input"]["tokens"] == 0
		assert result["output"]["tokens"] == 0
		assert result["total"]["tokens"] == 0

	def test_usage_calculator_total_cost_sum(self, sample_llm_response):
		"""Test that total cost is sum of input and output costs."""
		config = OmegaConf.create({"model": "gpt-4o"})
		calculator = UsageCalculator(config, sample_llm_response)

		result = calculator.calculate()

		expected_total_cost = result["input"]["cost"] + result["output"]["cost"]
		assert abs(result["total"]["cost"] - expected_total_cost) < 0.0001


class TestUrlSourceInput:
	"""Test URL ingestion and extraction flow."""

	def test_url_source_input_success(self, monkeypatch):
		"""Should return public URL-source output and metadata on success."""
		config = OmegaConf.create(
			{
				"url_prompt_folder": "prompts",
				"url_prompt_file": "url_source_input.jinja",
			}
		)

		def mock_fetch_url(_url):
			return "<html>mock content</html>"

		def mock_extract(*_args, **_kwargs):
			return (
				'{"title":"Mock CTI","author":"Analyst","date":"2026-01-01",'
				'"text":"APT group used malware.\\n\\nSubscribe\\nAPT group used malware."}'
			)

		monkeypatch.setattr("ctinexus.llm_processor.trafilatura.fetch_url", mock_fetch_url)
		monkeypatch.setattr("ctinexus.llm_processor.trafilatura.extract", mock_extract)

		result = UrlSourceInput(config).call("example.com/report")

		assert result["status"] == "success"
		assert result["url"] == "https://example.com/report"
		assert result["metadata"]["title"] == "Mock CTI"
		assert result["source_domain"] == "example.com"
		assert "Subscribe" not in result["summarized_text"]
		assert "Subscribe" not in result["final_text"]
		assert result["final_text"]
		assert "normalized_text" not in result
		assert "normalized_text_length" not in result
		assert "focused_text" not in result
		assert "focused_text_length" not in result
		assert "extraction_candidates" not in result

	def test_url_source_input_invalid_url(self):
		"""Should fail with invalid_url for malformed input."""
		config = OmegaConf.create({})
		result = UrlSourceInput(config).call("not a valid url value")
		assert result["status"] == "error"
		assert result["error"]["code"] == "invalid_url"

	def test_url_source_input_fetch_failure(self, monkeypatch):
		"""Should fail with fetch_failed when fetch_url returns no content."""
		config = OmegaConf.create({})
		monkeypatch.setattr("ctinexus.llm_processor.trafilatura.fetch_url", lambda *_args, **_kwargs: None)
		result = UrlSourceInput(config).call("https://example.com/report")
		assert result["status"] == "error"
		assert result["error"]["code"] == "fetch_failed"

	def test_url_source_input_extraction_failure(self, monkeypatch):
		"""Should fail with extraction_failed when extract returns no text."""
		config = OmegaConf.create({})
		monkeypatch.setattr("ctinexus.llm_processor.trafilatura.fetch_url", lambda *_args, **_kwargs: "<html />")
		monkeypatch.setattr("ctinexus.llm_processor.trafilatura.extract", lambda *_args, **_kwargs: None)

		result = UrlSourceInput(config).call("https://example.com/report")
		assert result["status"] == "error"
		assert result["error"]["code"] == "extraction_failed"
