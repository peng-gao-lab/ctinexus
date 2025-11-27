"""Unit tests for ctinexus.utils.path_utils module."""

import os

from ctinexus.utils.path_utils import BASE_DIR, resolve_path


class TestPathUtils:
	"""Test path utility functions."""

	def test_base_dir_exists(self):
		"""Test that BASE_DIR points to a valid directory."""
		assert os.path.isdir(BASE_DIR)

	def test_base_dir_is_ctinexus(self):
		"""Test that BASE_DIR points to the ctinexus package directory."""
		assert BASE_DIR.endswith("ctinexus")

	def test_resolve_path_single_component(self):
		"""Test resolving a single path component."""
		result = resolve_path("config")
		assert os.path.isabs(result)
		assert result.endswith(os.path.join("ctinexus", "config"))

	def test_resolve_path_multiple_components(self):
		"""Test resolving multiple path components."""
		result = resolve_path("config", "config.yaml")
		assert os.path.isabs(result)
		assert result.endswith(os.path.join("ctinexus", "config", "config.yaml"))

	def test_resolve_path_returns_absolute(self):
		"""Test that resolve_path always returns an absolute path."""
		result = resolve_path("test", "path")
		assert os.path.isabs(result)

	def test_resolve_path_with_nested_dirs(self):
		"""Test resolving nested directory paths."""
		result = resolve_path("data", "annotation", "test.json")
		assert result.endswith(os.path.join("ctinexus", "data", "annotation", "test.json"))

	def test_resolve_path_empty_args(self):
		"""Test resolve_path with no arguments returns BASE_DIR."""
		result = resolve_path()
		assert result == BASE_DIR

	def test_config_file_exists(self):
		"""Test that the config file can be resolved and exists."""
		config_path = resolve_path("config", "config.yaml")
		assert os.path.isfile(config_path)

	def test_cost_json_exists(self):
		"""Test that cost.json can be resolved and exists."""
		cost_path = resolve_path("config", "cost.json")
		assert os.path.isfile(cost_path)

	def test_prompts_directory_exists(self):
		"""Test that prompts directory can be resolved and exists."""
		prompts_path = resolve_path("prompts")
		assert os.path.isdir(prompts_path)
