import json
import os

from .app import get_default_models_for_provider
from .graph_constructor import create_graph_visualization
from .utils.gradio_utils import run_pipeline
from .utils.model_utils import MODELS, check_api_key


def process_cti_report(
	text: str,
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
	entity typing, entity alignment, and link prediction. It also generates an entity-relation graph visualization,
	optionally writes results to a file, and exports the final graph to Neo4j.
	"""
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
		ie_model=ie_model_full,
		et_model=et_model_full,
		ea_model=ea_model_full,
		lp_model=lp_model_full,
		similarity_threshold=similarity_threshold,
	)
	if isinstance(result, str) and result.startswith("Error:"):
		raise RuntimeError(result)

	# Parse pipeline output
	result_dict = json.loads(result)
	# Generate visualization (optional for comparison)
	graph_filepath = create_graph_visualization(result_dict)
	result_dict["entity_relation_graph"] = graph_filepath

	# Write output file if specified
	if output:
		output_dir = os.path.dirname(output)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		with open(output, "w", encoding="utf-8") as f:
			json.dump(result_dict, f, indent=4)

	return result_dict
