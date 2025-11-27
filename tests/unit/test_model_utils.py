"""Unit tests for ctinexus.utils.model_utils module."""

from ctinexus.utils.model_utils import (
	EMBEDDING_MODELS,
	MODELS,
	check_api_key,
	get_embedding_model_choices,
	get_model_choices,
	get_model_provider,
)


class TestCheckApiKey:
	"""Test API key checking functionality."""

	def test_check_api_key_with_openai(self, monkeypatch):
		"""Test that OpenAI models are registered when API key is present."""
		monkeypatch.setenv("OPENAI_API_KEY", "test-key")
		# Clear existing models
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is True
		assert "OpenAI" in MODELS
		assert "gpt-4o" in MODELS["OpenAI"]
		assert "OpenAI" in EMBEDDING_MODELS
		assert "text-embedding-3-large" in EMBEDDING_MODELS["OpenAI"]

	def test_check_api_key_with_gemini(self, monkeypatch):
		"""Test that Gemini models are registered when API key is present."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		monkeypatch.setenv("GEMINI_API_KEY", "test-key")
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is True
		assert "Gemini" in MODELS
		assert "gemini-2.0-flash" in MODELS["Gemini"]
		assert "Gemini" in EMBEDDING_MODELS
		assert "gemini-embedding-001" in EMBEDDING_MODELS["Gemini"]

	def test_check_api_key_with_aws(self, monkeypatch):
		"""Test that AWS models are registered when credentials are present."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		monkeypatch.delenv("GEMINI_API_KEY", raising=False)
		monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is True
		assert "AWS" in MODELS
		assert "anthropic.claude-3-5-sonnet" in MODELS["AWS"]
		assert "AWS" in EMBEDDING_MODELS
		assert "amazon.titan-embed-text-v2:0" in EMBEDDING_MODELS["AWS"]

	def test_check_api_key_with_ollama(self, monkeypatch):
		"""Test that Ollama models are registered when base URL is set."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		monkeypatch.delenv("GEMINI_API_KEY", raising=False)
		monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
		monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is True
		assert "Ollama" in MODELS
		assert "llama3.1:8b" in MODELS["Ollama"]
		assert "Ollama" in EMBEDDING_MODELS
		assert "nomic-embed-text" in EMBEDDING_MODELS["Ollama"]

	def test_check_api_key_no_keys(self, monkeypatch):
		"""Test that check returns False when no API keys are set."""
		monkeypatch.delenv("OPENAI_API_KEY", raising=False)
		monkeypatch.delenv("GEMINI_API_KEY", raising=False)
		monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
		monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is False
		assert len(MODELS) == 0

	def test_check_api_key_multiple_providers(self, monkeypatch):
		"""Test that multiple providers can be registered simultaneously."""
		monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
		monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
		MODELS.clear()
		EMBEDDING_MODELS.clear()

		result = check_api_key()

		assert result is True
		assert "OpenAI" in MODELS
		assert "Gemini" in MODELS


class TestGetModelProvider:
	"""Test model provider detection."""

	def test_get_provider_from_full_model_string(self, mock_env_vars):
		"""Test extracting provider from 'provider/model' format."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		provider = get_model_provider("OpenAI/gpt-4o", None)
		assert provider == "OpenAI"

	def test_get_provider_from_embedding_string(self, mock_env_vars):
		"""Test extracting provider from embedding model string."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		provider = get_model_provider(None, "Gemini/gemini-embedding-001")
		assert provider == "Gemini"

	def test_get_provider_from_model_name(self, mock_env_vars):
		"""Test finding provider from model name."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		provider = get_model_provider("gpt-4o", None)
		assert provider == "OpenAI"

	def test_get_provider_from_embedding_name(self, mock_env_vars):
		"""Test finding provider from embedding model name."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		provider = get_model_provider(None, "text-embedding-3-large")
		assert provider == "OpenAI"

	def test_get_provider_unknown_model(self, mock_env_vars):
		"""Test that unknown models return None."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		provider = get_model_provider("unknown-model", None)
		assert provider is None

	def test_get_provider_both_none(self, mock_env_vars):
		"""Test with both parameters as None."""
		provider = get_model_provider(None, None)
		assert provider is None


class TestGetModelChoices:
	"""Test model choices retrieval."""

	def test_get_model_choices_valid_provider(self, mock_env_vars):
		"""Test getting model choices for a valid provider."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		choices = get_model_choices("OpenAI")

		assert len(choices) > 0
		assert all(isinstance(choice, tuple) for choice in choices)
		assert all(len(choice) == 2 for choice in choices)
		# Check that one of the expected models is present
		model_keys = [key for _, key in choices]
		assert "gpt-4o" in model_keys

	def test_get_model_choices_invalid_provider(self):
		"""Test getting model choices for an invalid provider."""
		choices = get_model_choices("InvalidProvider")
		assert choices == []

	def test_get_model_choices_none_provider(self):
		"""Test getting model choices with None provider."""
		choices = get_model_choices(None)
		assert choices == []

	def test_get_model_choices_format(self, mock_env_vars):
		"""Test that model choices have correct format (description, key)."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		choices = get_model_choices("OpenAI")

		for description, key in choices:
			assert isinstance(description, str)
			assert isinstance(key, str)
			assert len(description) > 0
			assert len(key) > 0


class TestGetEmbeddingModelChoices:
	"""Test embedding model choices retrieval."""

	def test_get_embedding_choices_valid_provider(self, mock_env_vars):
		"""Test getting embedding choices for a valid provider."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		choices = get_embedding_model_choices("OpenAI")

		assert len(choices) > 0
		assert all(isinstance(choice, tuple) for choice in choices)
		# Check that expected embedding models are present
		model_keys = [key for _, key in choices]
		assert "text-embedding-3-large" in model_keys

	def test_get_embedding_choices_invalid_provider(self):
		"""Test getting embedding choices for an invalid provider."""
		choices = get_embedding_model_choices("InvalidProvider")
		assert choices == []

	def test_get_embedding_choices_none_provider(self):
		"""Test getting embedding choices with None provider."""
		choices = get_embedding_model_choices(None)
		assert choices == []

	def test_get_embedding_choices_all_providers(self, mock_env_vars):
		"""Test that all providers with API keys have embedding models."""
		MODELS.clear()
		EMBEDDING_MODELS.clear()
		check_api_key()

		for provider in EMBEDDING_MODELS.keys():
			choices = get_embedding_model_choices(provider)
			assert len(choices) > 0
