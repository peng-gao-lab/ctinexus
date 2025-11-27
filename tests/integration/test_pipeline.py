"""Integration tests for CTINexus pipeline."""

import json

from ctinexus import process_cti_report


class TestProcessCTIReport:
	"""Integration tests for the main process_cti_report function."""

	def test_process_cti_report_with_mocked_llm(self, sample_cti_text, mock_env_vars, mocker, tmp_path):
		"""Test full pipeline with mocked LLM calls."""
		# Mock LLM responses
		mock_completion = mocker.patch("ctinexus.llm_processor.litellm.completion")
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")

		# Mock completion response
		class MockResponse:
			def __init__(self):
				self.id = "test-123"
				self.choices = [MockChoice()]
				self.usage = MockUsage()

		class MockChoice:
			def __init__(self):
				self.message = MockMessage()

		class MockMessage:
			def __init__(self):
				self.content = json.dumps(
					{
						"triplets": [
							{
								"subject": {"text": "APT29", "class": "Malware"},
								"relation": "uses",
								"object": {"text": "PowerShell", "class": "Tool"},
							}
						]
					}
				)

		class MockUsage:
			def __init__(self):
				self.prompt_tokens = 100
				self.completion_tokens = 50

		mock_completion.return_value = MockResponse()

		# Mock embedding response
		mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

		# Change to temp directory for output
		import os

		original_dir = os.getcwd()
		os.chdir(tmp_path)

		try:
			result = process_cti_report(
				text=sample_cti_text,
				provider="OpenAI",
				model="gpt-4o",
				output=str(tmp_path / "output.json"),
			)

			# Verify result structure
			assert "text" in result
			assert "IE" in result
			assert "ET" in result
			assert "EA" in result
			assert "LP" in result
			assert "entity_relation_graph" in result

			# Verify output file was created
			output_file = tmp_path / "output.json"
			assert output_file.exists()

			# Verify graph file was created
			assert result["entity_relation_graph"].endswith(".html")

		finally:
			os.chdir(original_dir)

	def test_process_cti_report_no_api_key(self, sample_cti_text, monkeypatch):
		"""Test that process_cti_report raises error when no API keys are configured."""
		# Remove all API keys
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		monkeypatch.delenv("GEMINI_API_KEY", raising=False)
		monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
		monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

		# Clear MODELS dict
		from ctinexus.utils.model_utils import MODELS

		MODELS.clear()

		import pytest

		with pytest.raises(RuntimeError, match="No API Keys Configured"):
			process_cti_report(text=sample_cti_text)

	def test_process_cti_report_invalid_provider(self, sample_cti_text, mock_env_vars):
		"""Test that invalid provider raises ValueError."""
		import pytest

		with pytest.raises(ValueError, match="Provider .* not available"):
			process_cti_report(text=sample_cti_text, provider="InvalidProvider")

	def test_process_cti_report_auto_detect_provider(self, sample_cti_text, mock_env_vars, mocker, tmp_path):
		"""Test that provider is auto-detected when not specified."""
		# Mock LLM calls
		mock_completion = mocker.patch("ctinexus.llm_processor.litellm.completion")
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")

		class MockResponse:
			def __init__(self):
				self.id = "test-123"
				self.choices = [MockChoice()]
				self.usage = MockUsage()

		class MockChoice:
			def __init__(self):
				self.message = MockMessage()

		class MockMessage:
			def __init__(self):
				self.content = json.dumps(
					{
						"triplets": [
							{
								"subject": {"text": "APT29", "class": "Malware"},
								"relation": "uses",
								"object": {"text": "PowerShell", "class": "Tool"},
							}
						]
					}
				)

		class MockUsage:
			def __init__(self):
				self.prompt_tokens = 100
				self.completion_tokens = 50

		mock_completion.return_value = MockResponse()
		mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

		import os

		original_dir = os.getcwd()
		os.chdir(tmp_path)

		try:
			# Don't specify provider - should auto-detect
			result = process_cti_report(text=sample_cti_text)

			assert "IE" in result
			assert "ET" in result
			assert "EA" in result

		finally:
			os.chdir(original_dir)


class TestPipelineComponents:
	"""Integration tests for individual pipeline components."""

	def test_ie_to_et_flow(self, mocker):
		"""Test flow from Intelligence Extraction to Entity Tagging."""
		from omegaconf import OmegaConf

		from ctinexus.llm_processor import LLMExtractor, LLMTagger

		# Mock LLM responses
		mock_completion = mocker.patch("ctinexus.llm_processor.litellm.completion")

		class MockResponse:
			def __init__(self, content_dict):
				self.id = "test-123"
				self.choices = [MockChoice(content_dict)]
				self.usage = MockUsage()

		class MockChoice:
			def __init__(self, content_dict):
				self.message = MockMessage(content_dict)

		class MockMessage:
			def __init__(self, content_dict):
				self.content = json.dumps(content_dict)

		class MockUsage:
			def __init__(self):
				self.prompt_tokens = 100
				self.completion_tokens = 50

		# First call returns IE results
		ie_response = MockResponse(
			{
				"triplets": [
					{"subject": "APT29", "relation": "uses", "object": "PowerShell"},
				]
			}
		)

		# Second call returns ET results
		et_response = MockResponse(
			{
				"tagged_triples": [
					{
						"subject": {"text": "APT29", "class": "Malware"},
						"relation": "uses",
						"object": {"text": "PowerShell", "class": "Tool"},
					}
				]
			}
		)

		mock_completion.side_effect = [ie_response, et_response]

		# Run IE
		ie_config = OmegaConf.create(
			{
				"model": "test-model",
				"provider": "test",
				"ie_templ": "ie.jinja",
				"ie_prompt_set": "prompts",
				"retriever": "fixed",
			}
		)
		extractor = LLMExtractor(ie_config)
		ie_result = extractor.call("APT29 uses PowerShell")

		assert "IE" in ie_result
		assert len(ie_result["IE"]["triplets"]) > 0

		# Run ET
		et_config = OmegaConf.create(
			{
				"model": "test-model",
				"provider": "test",
				"tag_prompt_folder": "prompts",
				"tag_prompt_file": "et.jinja",
			}
		)
		tagger = LLMTagger(et_config)
		et_result = tagger.call(ie_result)

		assert "ET" in et_result
		assert "typed_triplets" in et_result["ET"]

	def test_full_pipeline_data_flow(self, sample_cti_text, mock_env_vars, mocker, tmp_path):
		"""Test that data flows correctly through entire pipeline."""
		# This is a comprehensive integration test
		mock_completion = mocker.patch("ctinexus.llm_processor.litellm.completion")
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")

		class MockResponse:
			def __init__(self):
				self.id = "test-123"
				self.choices = [MockChoice()]
				self.usage = MockUsage()

		class MockChoice:
			def __init__(self):
				self.message = MockMessage()

		class MockMessage:
			def __init__(self):
				self.content = json.dumps(
					{
						"triplets": [
							{
								"subject": {"text": "APT29", "class": "Malware"},
								"relation": "uses",
								"object": {"text": "PowerShell", "class": "Tool"},
							}
						]
					}
				)

		class MockUsage:
			def __init__(self):
				self.prompt_tokens = 100
				self.completion_tokens = 50

		mock_completion.return_value = MockResponse()
		mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

		import os

		original_dir = os.getcwd()
		os.chdir(tmp_path)

		try:
			result = process_cti_report(text=sample_cti_text, provider="OpenAI")

			# Verify each stage produced expected output
			assert result["text"] == sample_cti_text

			# IE stage
			assert "triplets" in result["IE"]
			assert isinstance(result["IE"]["triplets"], list)

			# ET stage
			assert "typed_triplets" in result["ET"]

			# EA stage
			assert "aligned_triplets" in result["EA"]
			assert "entity_num" in result["EA"]

			# LP stage
			assert "predicted_links" in result["LP"]

			# Graph visualization
			assert "entity_relation_graph" in result

		finally:
			os.chdir(original_dir)
