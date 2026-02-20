import json
import os
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

try:
	__version__ = version("ctinexus")
except PackageNotFoundError:
	__version__ = "dev"

__all__ = [
	"__version__",
	"get_default_models_for_provider",
	"process_cti_report",
]


def _load_environment() -> None:
	load_dotenv()
	load_dotenv(os.path.join(os.getcwd(), ".env"))


def get_default_models_for_provider(provider):
	from .app import get_default_models_for_provider as _get_defaults

	return _get_defaults(provider)


def process_cti_report(
	text: str = None,
	source_url: str = None,
	provider: str = None,
	model: str = None,
	embedding_model: str = None,
	ie_model: str = None,
	et_model: str = None,
	ea_model: str = None,
	lp_model: str = None,
	similarity_threshold: float = 0.6,
	output: str = None,
) -> dict:
	"""
	Process a Cyber Threat Intelligence (CTI) report and return the results as a dictionary.

	This function processes a CTI report using various language models and embedding models for information extraction,
	entity typing, entity alignment, and link prediction. It also generates an entity-relation graph
	visualization and optionally writes the results to a specified output file.

	Args:
		text (str, optional): The raw text of the CTI report to process.
		source_url (str, optional): URL of a CTI report/blog to ingest and process.
		provider (str, optional): The name of the model provider. Defaults to the first available provider.
		model (str, optional): The base model to use for processing. Defaults to the provider's default model.
		embedding_model (str, optional): The embedding model to use. Defaults to the provider's default embedding model.
		ie_model (str, optional): The information extraction model to use. Defaults to the base model.
		et_model (str, optional): The entity typing model to use. Defaults to the base model.
		ea_model (str, optional): The entity alignment model to use. Defaults to the embedding model.
		lp_model (str, optional): The link prediction model to use. Defaults to the base model.
		similarity_threshold (float, optional): The threshold for similarity in entity alignment. Defaults to 0.6.
		output (str, optional): The file path to write the output JSON. If not provided, results are not saved to a file.

	Returns:
		dict: A dictionary containing the processed results, including the entity-relation graph file path.
	"""
	from .graph_constructor import create_graph_visualization
	from .utils.gradio_utils import run_pipeline
	from .utils.model_utils import MODELS, check_api_key

	if not text and not source_url:
		raise ValueError("No input provided. Provide either `text` or `source_url`.")
	if text and source_url:
		raise ValueError("`text` and `source_url` are mutually exclusive. Provide only one.")

	api_keys_available = check_api_key()
	if not api_keys_available:
		raise RuntimeError(
			"No API Keys Configured. Please provide one API key in the `.env` file from the supported providers."
		)

	available_providers = list(MODELS.keys())
	if provider:
		provider_matched = next((p for p in available_providers if provider.lower() == p.lower()), None)
		if not provider_matched:
			raise ValueError(f"Provider '{provider}' not available. Available providers: {available_providers}")
		provider = provider_matched
	else:
		if available_providers:
			provider = available_providers[0]
		else:
			raise RuntimeError("No API keys configured")

	defaults = get_default_models_for_provider(provider)
	base_model = model or defaults.get("model")
	base_embedding_model = embedding_model or defaults.get("embedding_model")

	ie_model_full = f"{provider}/{ie_model or base_model}"
	et_model_full = f"{provider}/{et_model or base_model}"
	ea_model_full = f"{provider}/{ea_model or base_embedding_model}"
	lp_model_full = f"{provider}/{lp_model or base_model}"

	result = run_pipeline(
		text=text,
		source_url=source_url,
		ie_model=ie_model_full,
		et_model=et_model_full,
		ea_model=ea_model_full,
		lp_model=lp_model_full,
		similarity_threshold=similarity_threshold,
	)
	if isinstance(result, str) and result.startswith("Error:"):
		raise RuntimeError(result)

	# Create Entity Relation Graph
	result_dict = json.loads(result)
	_, graph_filepath = create_graph_visualization(result_dict)
	result_dict["entity_relation_graph"] = graph_filepath

	# Write output if requested
	if output:
		output_dir = os.path.dirname(output)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		with open(output, "w", encoding="utf-8") as f:
			json.dump(result_dict, f, indent=4)

	return result_dict


_load_environment()
