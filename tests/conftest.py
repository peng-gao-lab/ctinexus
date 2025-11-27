"""Pytest configuration and shared fixtures for CTINexus tests."""

import json
from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture
def sample_cti_text() -> str:
	"""Sample CTI text for testing."""
	return """
	APT29 used PowerShell to download additional malware from command-and-control
	server at 192.168.1.100. The attack exploited CVE-2023-1234 in Microsoft Exchange.
	The malware, known as SUNBURST, was used to steal credentials and exfiltrate data.
	"""


@pytest.fixture
def sample_triplets() -> list:
	"""Sample triplets for testing."""
	return [
		{"subject": {"text": "APT29"}, "relation": "uses", "object": {"text": "PowerShell"}},
		{"subject": {"text": "APT29"}, "relation": "exploits", "object": {"text": "CVE-2023-1234"}},
		{
			"subject": {"text": "APT29"},
			"relation": "communicates_with",
			"object": {"text": "192.168.1.100"},
		},
	]


@pytest.fixture
def sample_ie_result() -> Dict:
	"""Sample Intelligence Extraction result."""
	return {
		"text": "APT29 used PowerShell to download malware.",
		"IE": {
			"triplets": [
				{
					"subject": {"text": "APT29", "class": "Malware"},
					"relation": "uses",
					"object": {"text": "PowerShell", "class": "Tool"},
				}
			],
			"triples_count": 1,
			"model_usage": {
				"model": "test-model",
				"input": {"tokens": 100, "cost": 0.001},
				"output": {"tokens": 50, "cost": 0.002},
				"total": {"tokens": 150, "cost": 0.003},
			},
			"response_time": 1.5,
			"Prompt": {"prompt_template": "ie.jinja", "demo_retriever": "fixed"},
		},
	}


@pytest.fixture
def sample_typed_triplets() -> list:
	"""Sample typed triplets for testing."""
	return [
		{
			"subject": {"text": "APT29", "class": "Malware"},
			"relation": "uses",
			"object": {"text": "PowerShell", "class": "Tool"},
		},
		{
			"subject": {"text": "APT29", "class": "Malware"},
			"relation": "exploits",
			"object": {"text": "CVE-2023-1234", "class": "Indicator"},
		},
	]


@pytest.fixture
def sample_aligned_triplets() -> list:
	"""Sample aligned triplets with entity information."""
	return [
		{
			"subject": {
				"mention_id": 0,
				"mention_text": "APT29",
				"mention_class": "Malware",
				"mention_merged": [],
				"entity_id": 0,
				"entity_text": "APT29",
			},
			"relation": "uses",
			"object": {
				"mention_id": 1,
				"mention_text": "PowerShell",
				"mention_class": "Tool",
				"mention_merged": [],
				"entity_id": 1,
				"entity_text": "PowerShell",
			},
		}
	]


@pytest.fixture
def sample_ea_result(sample_ie_result, sample_aligned_triplets) -> Dict:
	"""Sample Entity Alignment result."""
	result = sample_ie_result.copy()
	result["ET"] = {
		"typed_triplets": [
			{
				"subject": {"text": "APT29", "class": "Malware"},
				"relation": "uses",
				"object": {"text": "PowerShell", "class": "Tool"},
			}
		],
		"response_time": 1.0,
		"model_usage": {
			"model": "test-model",
			"input": {"tokens": 50, "cost": 0.0005},
			"output": {"tokens": 30, "cost": 0.001},
			"total": {"tokens": 80, "cost": 0.0015},
		},
	}
	result["EA"] = {
		"aligned_triplets": sample_aligned_triplets,
		"entity_num": 2,
		"mentions_num": 2,
		"model_usage": {
			"model": "test-embedding-model",
			"input": {"tokens": 20, "cost": 0.0001},
			"output": {"tokens": 0, "cost": 0},
			"total": {"tokens": 20, "cost": 0.0001},
		},
		"response_time": 0.5,
	}
	return result


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
	"""Temporary output directory for test artifacts."""
	output_dir = tmp_path / "ctinexus_output"
	output_dir.mkdir(exist_ok=True)
	return output_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
	"""Mock environment variables for testing."""
	monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
	monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
	monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-aws-key")
	monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-aws-secret")
	monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture
def sample_llm_response():
	"""Sample LLM response object."""

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
							"subject": "APT29",
							"relation": "uses",
							"object": "PowerShell",
						}
					]
				}
			)

	class MockUsage:
		def __init__(self):
			self.prompt_tokens = 100
			self.completion_tokens = 50

	return MockResponse()


@pytest.fixture(autouse=True)
def clean_output_dir():
	"""Clean up output directory after each test."""
	yield
	# Cleanup happens after test
	output_dir = Path.cwd() / "ctinexus_output"
	if output_dir.exists():
		for file in output_dir.glob("network_*.html"):
			try:
				file.unlink()
			except Exception:
				pass
