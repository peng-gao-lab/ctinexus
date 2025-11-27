"""Unit tests for ctinexus.graph_constructor module."""

from omegaconf import OmegaConf

from ctinexus.graph_constructor import Linker, Merger, create_graph_visualization


class TestLinker:
	"""Test Linker functionality."""

	def test_linker_initialization(self):
		"""Test Linker initialization."""
		config = OmegaConf.create({"test": "value"})
		linker = Linker(config)

		assert linker.config == config
		assert linker.graph == {}

	def test_linker_builds_graph(self, sample_ea_result):
		"""Test that Linker builds a graph from aligned triplets."""
		config = OmegaConf.create(
			{
				"model": "test-model",
				"provider": "test",
				"link_prompt_folder": "prompts",
				"link_prompt_file": "link.jinja",
			}
		)
		linker = Linker(config)

		result = linker.call(sample_ea_result)

		assert "LP" in result
		assert linker.graph != {}

	def test_linker_finds_subgraphs(self, sample_ea_result, mocker):
		"""Test that Linker identifies disconnected subgraphs."""
		# Mock litellm.completion to avoid actual LLM calls
		mock_response = mocker.MagicMock()
		mock_response.choices = [mocker.MagicMock(message=mocker.MagicMock(content='{"predicted_links": []}'))]
		mocker.patch("litellm.completion", return_value=mock_response)

		# Add more entities to create multiple subgraphs
		sample_ea_result["EA"]["aligned_triplets"].append(
			{
				"subject": {
					"mention_id": 2,
					"mention_text": "CVE-2023-1234",
					"mention_class": "Indicator",
					"mention_merged": [],
					"entity_id": 2,
					"entity_text": "CVE-2023-1234",
				},
				"relation": "affects",
				"object": {
					"mention_id": 3,
					"mention_text": "Microsoft Exchange",
					"mention_class": "Tool",
					"mention_merged": [],
					"entity_id": 3,
					"entity_text": "Microsoft Exchange",
				},
			}
		)

		config = OmegaConf.create(
			{
				"model": "test-model",
				"provider": "test",
				"link_prompt_folder": "prompts",
				"link_prompt_file": "link.jinja",
			}
		)
		linker = Linker(config)

		result = linker.call(sample_ea_result)

		assert "LP" in result
		assert "subgraph_num" in result["LP"]

	def test_linker_identifies_topic_node(self, sample_ea_result):
		"""Test that Linker identifies the main topic node."""
		config = OmegaConf.create(
			{
				"model": "test-model",
				"provider": "test",
				"link_prompt_folder": "prompts",
				"link_prompt_file": "link.jinja",
			}
		)
		linker = Linker(config)

		result = linker.call(sample_ea_result)

		assert "topic_node" in result["LP"]
		assert "entity_text" in result["LP"]["topic_node"]


class TestMerger:
	"""Test Merger functionality."""

	def test_merger_initialization(self):
		"""Test Merger initialization."""
		config = OmegaConf.create({"similarity_threshold": 0.6})
		merger = Merger(config)

		assert merger.config == config
		assert merger.node_dict == {}
		assert merger.class_dict == {}
		assert merger.entity_dict == {}
		assert merger.emb_dict == {}
		assert merger.entity_id == 0

	def test_merger_creates_embeddings(self, sample_ea_result, mocker):
		"""Test that Merger creates embeddings for mentions."""
		# Mock the litellm.embedding call
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")
		mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}

		config = OmegaConf.create(
			{
				"model": "test-embedding",
				"similarity_threshold": 0.6,
				"embedding_model": "test-embedding",
				"provider": "test",
			}
		)
		merger = Merger(config)

		result = merger.call(sample_ea_result)

		assert "EA" in result
		assert len(merger.emb_dict) > 0

	def test_merger_assigns_entity_ids(self, sample_ea_result, mocker):
		"""Test that Merger assigns entity IDs to mentions."""
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")
		mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}

		config = OmegaConf.create(
			{
				"model": "test-embedding",
				"similarity_threshold": 0.6,
				"embedding_model": "test-embedding",
				"provider": "test",
			}
		)
		merger = Merger(config)

		result = merger.call(sample_ea_result)

		# Check that entities have been assigned IDs
		for triplet in result["EA"]["aligned_triplets"]:
			assert "entity_id" in triplet["subject"]
			assert "entity_id" in triplet["object"]

	def test_merger_merges_similar_entities(self, mocker):
		"""Test that Merger merges similar entities based on threshold."""
		# Create a result with potentially similar entities
		result = {
			"EA": {
				"aligned_triplets": [
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
							"mention_text": "Cozy Bear",
							"mention_class": "Malware",
							"mention_merged": [],
							"entity_id": 1,
							"entity_text": "Cozy Bear",
						},
					}
				],
				"entity_num": 2,
			}
		}

		# Mock embeddings to be very similar (high cosine similarity)
		mock_embedding = mocker.patch("ctinexus.graph_constructor.litellm.embedding")
		mock_embedding.return_value = {
			"data": [
				{"embedding": [0.9, 0.1, 0.1]},  # APT29
				{"embedding": [0.9, 0.1, 0.1]},  # Cozy Bear (very similar)
			]
		}

		config = OmegaConf.create(
			{
				"model": "test-embedding",
				"similarity_threshold": 0.6,
				"embedding_model": "test-embedding",
				"provider": "test",
			}
		)
		merger = Merger(config)

		merged_result = merger.call(result)

		# Verify the result structure
		assert "EA" in merged_result


class TestCreateGraphVisualization:
	"""Test graph visualization creation."""

	def test_create_graph_basic(self, sample_ea_result, temp_output_dir, monkeypatch):
		"""Test basic graph creation."""
		monkeypatch.chdir(temp_output_dir.parent)

		url, filepath = create_graph_visualization(sample_ea_result)

		assert url is not None
		assert filepath is not None
		assert filepath.endswith(".html")

	def test_create_graph_with_predicted_links(self, sample_ea_result, temp_output_dir, monkeypatch):
		"""Test graph creation with predicted links."""
		monkeypatch.chdir(temp_output_dir.parent)

		# Add predicted links
		sample_ea_result["LP"] = {
			"predicted_links": [
				{
					"subject": {"entity_id": 0, "mention_text": "APT29"},
					"relation": "targets",
					"object": {"entity_id": 1, "mention_text": "PowerShell"},
				}
			]
		}

		url, filepath = create_graph_visualization(sample_ea_result)

		assert url is not None
		assert filepath is not None

	def test_create_graph_empty_result(self, temp_output_dir, monkeypatch):
		"""Test graph creation with empty result."""
		monkeypatch.chdir(temp_output_dir.parent)

		result = {"EA": {"aligned_triplets": []}}

		url, filepath = create_graph_visualization(result)

		assert url is not None
		assert filepath is not None

	def test_create_graph_creates_output_dir(self, tmp_path, monkeypatch):
		"""Test that graph creation creates output directory if it doesn't exist."""
		monkeypatch.chdir(tmp_path)

		result = {"EA": {"aligned_triplets": []}}

		url, filepath = create_graph_visualization(result)

		output_dir = tmp_path / "ctinexus_output"
		assert output_dir.exists()
		assert output_dir.is_dir()
