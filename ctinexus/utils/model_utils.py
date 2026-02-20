import os

from dotenv import load_dotenv

load_dotenv()

# Available models
MODELS = {}
EMBEDDING_MODELS = {}


def check_api_key() -> bool:
	"""Define Models and check if API KEYS are set"""
	if os.getenv("OPENAI_API_KEY"):
		MODELS["OpenAI"] = {
			"gpt-4.1-mini": "GPT-4.1 Mini — Balanced for intelligence, speed, and cost ($0.4 • $1.6)",
			"gpt-4.1": "GPT-4.1 — Flagship GPT model for complex tasks ($2 • $8)",
			"o4-mini": "o4 Mini — Faster, more affordable reasoning model ($1.1 • $4.4)",
			"o3-mini": "o3 Mini — A small reasoning model alternative to o3 ($1.1 • $4.4)",
			"o3": "o3 — Most powerful reasoning model ($2 • $8)",
			"o3-pro": "o3 Pro — Version of o3 with more compute for better responses ($20 • $80)",
			"gpt-4o": "GPT-4o — Fast, intelligent, flexible GPT model ($2.5 • $10)",
			"gpt-4": "GPT-4 — An older high-intelligence GPT model ($30 • $60)",
			"gpt-4-turbo": "GPT-4 Turbo — An older high-intelligence GPT model ($10 • $30)",
			"gpt-3.5-turbo": "GPT-3.5 Turbo — Legacy GPT model for cheaper chat and non-chat tasks ($0.5 • $1.5)",
			"gpt-4o-mini": "GPT-4o Mini — Fast, affordable small model for focused tasks ($0.15 • $0.6)",
			"gpt-4.1-nano": "GPT-4.1 Nano — Fastest, most cost-effective GPT-4.1 model ($0.1 • $0.4)",
		}
		EMBEDDING_MODELS["OpenAI"] = {
			"text-embedding-3-large": "Text Embedding 3 Large — Most capable embedding model ($0.13)",
			"text-embedding-3-small": "Text Embedding 3 Small — Small embedding model ($0.02)",
			"text-embedding-ada-002": "Text Embedding Ada 002 — Older embedding model ($0.1)",
		}

	if os.getenv("GEMINI_API_KEY"):
		MODELS["Gemini"] = {
			"gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite — Most cost-efficient for high throughput ($0.10 • $0.40)",
			"gemini-2.0-flash": "Gemini 2.0 Flash — Balanced multimodal model for agents ($0.10 • $0.40)",
			"gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite — Smallest, most cost-effective ($0.075 • $0.30)",
		}
		EMBEDDING_MODELS["Gemini"] = {
			"gemini-embedding-001": "Gemini Embedding — Text embeddings for relatedness ($0.15)",
		}

	if os.getenv("AWS_ACCESS_KEY_ID"):
		MODELS["AWS"] = {
			"anthropic.claude-3-7-sonnet": "Claude 3.7 Sonnet — Advanced reasoning for complex text tasks ($3 • $15)",
			"anthropic.claude-3-5-sonnet": "Claude 3.5 Sonnet — Balanced for intelligence and efficiency in text ($3 • $15)",
			"anthropic.claude-3-5-haiku": "Claude 3.5 Haiku — Fast, cost-effective for simple text tasks ($0.8 • $4)",
			"anthropic.claude-3-haiku": "Claude 3 Haiku — Fast, cost-effective for simple text tasks ($0.25 • $1.25)",
			"amazon.nova-micro-v1:0": "Nova Micro — Text-only, ultra-fast for chat and summarization ($0.035 • $0.14)",
			"amazon.nova-lite-v1:0": "Nova Lite — Multimodal, large context for complex text ($0.06 • $0.24)",
			"amazon.nova-pro-v1:0": "Nova Pro — High-performance multimodal for advanced text ($0.45 • $1.8)",
			"deepseek.r1-v1:0": "DeepSeek R1 — Cost-efficient for research and text generation ($0.14 • $0.7)",
			"mistral.pixtral-large-2502-v1:0": "Pixtral Large — Multimodal, excels in visual-text tasks ($1 • $3)",
			"meta.llama3-1-8b-instruct-v1:0": "Llama 3.1 8B — Lightweight, efficient for basic text tasks ($0.15 • $0.6)",
			"meta.llama3-1-70b-instruct-v1:0": "Llama 3.1 70B — Balanced for complex text and coding ($0.75 • $3)",
			"meta.llama3-2-11b-instruct-v1:0": "Llama 3.2 11B — Compact, optimized for multilingual text ($0.2 • $0.8)",
			"meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B — Balanced for complex text and coding ($0.75 • $3)",
		}
		EMBEDDING_MODELS["AWS"] = {
			"amazon.titan-embed-text-v2:0": "Titan Embed Text 2 — Large embedding model ($0.12)",
		}

	if os.getenv("OLLAMA_BASE_URL"):
		MODELS["Ollama"] = {
			"llama3.1:8b": "Llama 3.1 8B — Balanced performance for general use (Free)",
			"llama3.1:70b": "Llama 3.1 70B — High-performance model for complex tasks (Free)",
			"llama3:8b": "Llama 3 8B — Reliable model for general purpose tasks (Free)",
			"mistral:7b": "Mistral 7B — Efficient model with good reasoning (Free)",
			"mixtral:8x7b": "Mixtral 8x7B — Mixture of experts model (Free)",
			"qwen2.5:7b": "Qwen2.5 7B — Chinese-optimized multilingual model (Free)",
			"qwen2.5:14b": "Qwen2.5 14B — Larger Chinese-optimized model (Free)",
			"phi3:14b": "Phi-3 14B — Microsoft's mid-size model (Free)",
			"gemma2:9b": "Gemma 2 9B — Google's open model (Free)",
			"gemma2:27b": "Gemma 2 27B — Google's larger open model (Free)",
		}
		EMBEDDING_MODELS["Ollama"] = {
			"nomic-embed-text": "Nomic Embed Text — High-quality text embeddings (Free)",
			"mxbai-embed-large": "MixedBread AI Large — Advanced embedding model (Free)",
			"all-minilm": "All-MiniLM-L6-v2 — Compact embedding model (Free)",
			"snowflake-arctic-embed": "Snowflake Arctic Embed — Retrieval-optimized embeddings (Free)",
		}

	return True if MODELS else False


def get_model_provider(model, embedding_model):
	# If the model is in the format "provider/model"
	if model and "/" in model:
		return model.split("/")[0]

	if embedding_model and "/" in embedding_model:
		return embedding_model.split("/")[0]

	for provider, models in MODELS.items():
		if model in models:
			return provider

	for provider, models in EMBEDDING_MODELS.items():
		if embedding_model in models:
			return provider
	return None


def get_model_choices(provider):
	"""Get model choices with descriptions for the dropdown"""
	if not provider or provider not in MODELS:
		return []
	return [(desc, key) for key, desc in MODELS[provider].items()]


def get_embedding_model_choices(provider):
	"""Get model choices with descriptions for the dropdown"""
	if not provider or provider not in EMBEDDING_MODELS:
		return []
	return [(desc, key) for key, desc in EMBEDDING_MODELS[provider].items()]
