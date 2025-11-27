"""Unit tests for ctinexus.cti_processor module."""

import pytest
from omegaconf import OmegaConf

from ctinexus.cti_processor import IOC_detect, PostProcessor, preprocessor


class TestIOCDetect:
	"""Test IOC detection functionality."""

	@pytest.mark.parametrize(
		"text, expected_ioc",
		[
			("Server at 192.168.1.100", "192.168.1.100"),
			("CVE-2023-1234", "CVE-2023-1234"),
			("malicious.example.com", "malicious.example.com"),
			("http://malicious.com/payload", "http://malicious.com/payload"),
			("attacker@malicious.com", "attacker@malicious.com"),
			("5d41402abc4b2a76b9719d911017c592", "5d41402abc4b2a76b9719d911017c592"),
			(
				"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
				"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			),
			("Attack occurred in January 2023", "January 2023"),
			("version 1.2.3", "version 1.2.3"),
		],
	)
	def test_detect_single_iocs(self, text, expected_ioc):
		"""Test detection of single IOCs using parameterization."""
		iocs = IOC_detect([], text)
		assert expected_ioc in iocs

	def test_detect_multiple_ips(self):
		"""Test detection of multiple IP addresses."""
		iocs = IOC_detect(["10.0.0.1"], "192.168.1.100")
		assert len(iocs) == 2
		assert "10.0.0.1" in iocs
		assert "192.168.1.100" in iocs

	def test_no_iocs_in_normal_text(self):
		"""Test that normal text doesn't produce false IOC detections."""
		iocs = IOC_detect([], "This is normal text without indicators")
		assert len(iocs) == 0

	def test_detect_mixed_iocs(self):
		"""Test detection of multiple types of IOCs."""
		text = "APT29 at 192.168.1.1 exploited CVE-2023-1234 via malicious.com"
		iocs = IOC_detect([], text)
		assert "192.168.1.1" in iocs
		assert "CVE-2023-1234" in iocs
		assert "malicious.com" in iocs

	def test_empty_input(self):
		"""Test with empty input."""
		iocs = IOC_detect([], "")
		assert len(iocs) == 0

	def test_detect_from_merged_mentions(self):
		"""Test IOC detection from merged mentions list."""
		merged = ["192.168.1.1", "CVE-2023-1234"]
		iocs = IOC_detect(merged, "malicious.com")
		assert len(iocs) == 3


class TestPreprocessor:
	"""Test preprocessor functionality."""

	def test_preprocessor_basic(self, sample_ie_result):
		"""Test basic preprocessing of results."""
		# Add ET section which is expected by preprocessor
		sample_ie_result["ET"] = {"typed_triplets": sample_ie_result["IE"]["triplets"]}
		result = preprocessor(sample_ie_result)

		assert "EA" in result
		assert "aligned_triplets" in result["EA"]
		assert "mentions_num" in result["EA"]

	def test_preprocessor_creates_mention_ids(self, sample_ie_result):
		"""Test that preprocessor creates unique mention IDs."""
		# Add IE and ET sections
		sample_ie_result["ET"] = {
			"typed_triplets": [
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
		}

		result = preprocessor(sample_ie_result)

		# Check that mention_ids are assigned
		for triplet in result["EA"]["aligned_triplets"]:
			assert "mention_id" in triplet["subject"]
			assert "mention_id" in triplet["object"]

	def test_preprocessor_same_text_same_id(self, sample_ie_result):
		"""Test that same mention text gets same mention ID."""
		sample_ie_result["ET"] = {
			"typed_triplets": [
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
		}

		result = preprocessor(sample_ie_result)

		# Both subjects should have the same mention_id (both are "APT29")
		triplets = result["EA"]["aligned_triplets"]
		assert triplets[0]["subject"]["mention_id"] == triplets[1]["subject"]["mention_id"]

	def test_preprocessor_renames_fields(self, sample_ie_result):
		"""Test that preprocessor renames text to mention_text."""
		sample_ie_result["ET"] = {
			"typed_triplets": [
				{
					"subject": {"text": "APT29", "class": "Malware"},
					"relation": "uses",
					"object": {"text": "PowerShell", "class": "Tool"},
				}
			]
		}

		result = preprocessor(sample_ie_result)

		triplet = result["EA"]["aligned_triplets"][0]
		assert "mention_text" in triplet["subject"]
		assert triplet["subject"]["mention_text"] == "APT29"
		assert "text" not in triplet["subject"]

	def test_preprocessor_handles_dict_class(self, sample_ie_result):
		"""Test that preprocessor handles mention_class as dictionary."""
		sample_ie_result["ET"] = {
			"typed_triplets": [
				{
					"subject": {"text": "APT29", "class": {"Malware": 0.9}},
					"relation": "uses",
					"object": {"text": "PowerShell", "class": "Tool"},
				}
			]
		}

		result = preprocessor(sample_ie_result)

		triplet = result["EA"]["aligned_triplets"][0]
		assert triplet["subject"]["mention_class"] == "Malware"

	def test_preprocessor_empty_result(self):
		"""Test preprocessor with empty result."""
		result = preprocessor({})
		assert result == {}

	def test_preprocessor_counts_mentions(self, sample_ie_result):
		"""Test that preprocessor correctly counts unique mentions."""
		sample_ie_result["ET"] = {
			"typed_triplets": [
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
		}

		result = preprocessor(sample_ie_result)

		# Should have 3 unique mentions: APT29, PowerShell, CVE-2023-1234
		assert result["EA"]["mentions_num"] == 3


class TestPostProcessor:
	"""Test PostProcessor functionality."""

	def test_postprocessor_initialization(self):
		"""Test PostProcessor initialization."""
		config = OmegaConf.create({"test": "value"})
		processor = PostProcessor(config)

		assert processor.config == config
		assert processor.mention_dict == {}
		assert processor.node_dict == {}

	def test_postprocessor_call_preserves_structure(self, sample_ea_result):
		"""Test that PostProcessor preserves basic structure."""
		config = OmegaConf.create({})
		processor = PostProcessor(config)

		result = processor.call(sample_ea_result)

		assert "EA" in result
		assert "aligned_triplets" in result["EA"]

	def test_postprocessor_handles_empty_merged(self, sample_ea_result):
		"""Test PostProcessor with empty mention_merged."""
		config = OmegaConf.create({})
		processor = PostProcessor(config)

		# Ensure mention_merged is empty
		for triplet in sample_ea_result["EA"]["aligned_triplets"]:
			triplet["subject"]["mention_merged"] = []
			triplet["object"]["mention_merged"] = []

		result = processor.call(sample_ea_result)

		# Should complete without errors
		assert "EA" in result

	def test_postprocessor_processes_iocs(self):
		"""Test PostProcessor handles IOC detection in merged mentions."""
		config = OmegaConf.create({})
		processor = PostProcessor(config)

		result = {
			"EA": {
				"entity_num": 2,
				"aligned_triplets": [
					{
						"subject": {
							"mention_id": 0,
							"mention_text": "192.168.1.100",
							"mention_class": "Indicator",
							"mention_merged": ["10.0.0.1"],
							"entity_id": 0,
							"entity_text": "192.168.1.100",
						},
						"relation": "communicates_with",
						"object": {
							"mention_id": 1,
							"mention_text": "APT29",
							"mention_class": "Malware",
							"mention_merged": [],
							"entity_id": 1,
							"entity_text": "APT29",
						},
					}
				],
			}
		}

		processed = processor.call(result)

		# Should process without errors
		assert "EA" in processed

	def test_postprocessor_increments_entity_index(self):
		"""Test that PostProcessor can increment entity index."""
		config = OmegaConf.create({})
		processor = PostProcessor(config)

		result = {
			"EA": {
				"entity_num": 1,
				"aligned_triplets": [
					{
						"subject": {
							"mention_id": 0,
							"mention_text": "192.168.1.100",
							"mention_class": "Indicator",
							"mention_merged": ["10.0.0.1", "172.16.0.1"],
							"entity_id": 0,
							"entity_text": "192.168.1.100",
						},
						"relation": "test",
						"object": {
							"mention_id": 1,
							"mention_text": "APT29",
							"mention_class": "Malware",
							"mention_merged": [],
							"entity_id": 1,
							"entity_text": "APT29",
						},
					},
					{
						"subject": {
							"mention_id": 2,
							"mention_text": "10.0.0.1",
							"mention_class": "Indicator",
							"mention_merged": [],
							"entity_id": 2,
							"entity_text": "10.0.0.1",
						},
						"relation": "test",
						"object": {
							"mention_id": 1,
							"mention_text": "APT29",
							"mention_class": "Malware",
							"mention_merged": [],
							"entity_id": 1,
							"entity_text": "APT29",
						},
					},
				],
			}
		}

		processed = processor.call(result)
		assert "EA" in processed
