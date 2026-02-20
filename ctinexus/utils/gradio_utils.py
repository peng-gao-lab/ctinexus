import json
import traceback
from urllib.parse import urlparse

import gradio as gr
from hydra import compose, initialize
from omegaconf import DictConfig

from ..cti_processor import PostProcessor, preprocessor
from ..graph_constructor import Linker, Merger, create_graph_visualization
from ..llm_processor import LLMExtractor, LLMTagger, UrlSourceInput
from .model_utils import (
	MODELS,
	get_embedding_model_choices,
	get_model_choices,
	get_model_provider,
)
from .path_utils import resolve_path

CONFIG_PATH = "../config"


def get_metrics_box(
	ie_metrics: str = "",
	et_metrics: str = "",
	ea_metrics: str = "",
	lp_metrics: str = "",
):
	"""Generate metrics box HTML with optional metrics values"""
	return f'<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Intelligence Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Tagging</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Alignment</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td>{ie_metrics or ""}</td><td>{et_metrics or ""}</td><td>{ea_metrics or ""}</td><td>{lp_metrics or ""}</td></tr></table></div>'


def run_intel_extraction(config: DictConfig, text: str = None) -> dict:
	"""Wrapper for Intelligence Extraction"""
	return LLMExtractor(config).call(text)


def run_entity_tagging(config: DictConfig, result: dict) -> dict:
	"""Wrapper for Entity Tagging"""
	return LLMTagger(config).call(result)


def run_url_source_input(config: DictConfig, source_url: str) -> dict:
	"""Wrapper for URL source ingestion and extraction."""
	return UrlSourceInput(config).call(source_url)


def run_entity_alignment(config: DictConfig, result: dict) -> dict:
	"""Wrapper for Entity Alignment"""
	preprocessed_result = preprocessor(result)
	merged_result = Merger(config).call(preprocessed_result)
	final_result = PostProcessor(config).call(merged_result)
	return final_result


def run_link_prediction(config: DictConfig, result) -> dict:
	"""Wrapper for Link Prediction"""

	if not isinstance(result, dict):
		result = {"subgraphs": result}

	return Linker(config).call(result)


def get_config(model: str = None, embedding_model: str = None, similarity_threshold: float = 0.6) -> DictConfig:
	provider = get_model_provider(model, embedding_model)
	model = model.split("/")[-1] if model else None
	embedding_model = embedding_model.split("/")[-1] if embedding_model else None

	with initialize(version_base="1.2", config_path=CONFIG_PATH):
		overrides = []
		if model:
			overrides.append(f"model={model}")
		if embedding_model:
			overrides.append(f"embedding_model={embedding_model}")
		if similarity_threshold:
			overrides.append(f"similarity_threshold={similarity_threshold}")
		if provider:
			overrides.append(f"provider={provider}")
		config = compose(config_name="config.yaml", overrides=overrides)
	return config


def run_pipeline(
	text: str = None,
	source_url: str = None,
	ie_model: str = None,
	et_model: str = None,
	ea_model: str = None,
	lp_model: str = None,
	similarity_threshold: float = 0.6,
	progress=gr.Progress(track_tqdm=False),
):
	"""Run the entire pipeline in sequence"""
	if not text and not source_url:
		return "Error: Please enter CTI text or provide a report URL."

	if source_url and not is_valid_source_url(source_url):
		return "Error: Invalid URL format. Please provide a valid http/https URL."

	try:
		url_source_result = None

		if source_url and source_url.strip():
			config = get_config(ie_model, None, None)
			progress(0.05, desc="Ingesting URL source...")
			url_source_result = run_url_source_input(config, source_url)
			if url_source_result.get("status") != "success":
				error_info = url_source_result.get("error", {})
				error_code = error_info.get("code", "url_ingestion_failed")
				error_message = error_info.get("message", "URL ingestion failed.")
				return f"Error: [{error_code}] {error_message}"
			text = url_source_result.get("final_text") or url_source_result.get("normalized_text")

		if not text:
			return "Error: No usable report content was found from the URL source."

		config = get_config(ie_model, None, None)
		progress(0.2, desc="Intelligence Extraction...")
		extraction_result = run_intel_extraction(config, text)
		if url_source_result:
			extraction_result["URL_SOURCE"] = url_source_result

		config = get_config(et_model, None, None)
		progress(0.45, desc="Entity Tagging...")
		tagging_result = run_entity_tagging(config, extraction_result)

		progress(0.7, desc="Entity Alignment...")
		config = get_config(None, ea_model, similarity_threshold)
		config.similarity_threshold = similarity_threshold
		alignment_result = run_entity_alignment(config, tagging_result)

		config = get_config(lp_model, None, None)
		progress(0.9, desc="Link Prediction...")
		linking_result = run_link_prediction(config, alignment_result)

		progress(1.0, desc="Processing complete!")

		return json.dumps(linking_result, indent=4)
	except Exception as e:
		progress(1.0, desc="Error occurred!")
		traceback.print_exc()
		return f"Error: {str(e)}"


def process_and_visualize(
	input_source,
	text,
	source_url,
	ie_model,
	et_model,
	ea_model,
	lp_model,
	similarity_threshold,
	provider_dropdown=None,
	custom_model_input=None,
	custom_embedding_model_input=None,
	progress=gr.Progress(track_tqdm=False),
):
	if input_source == "CTI Report URL":
		text = None
	else:
		source_url = None

	# Apply custom model only to dropdowns where 'Other' is selected
	custom_model = f"{provider_dropdown}/{custom_model_input}" if provider_dropdown else custom_model_input
	custom_embedding_model = (
		f"{provider_dropdown}/{custom_embedding_model_input}" if provider_dropdown else custom_embedding_model_input
	)

	ie_model = custom_model if ie_model == "Other" else ie_model
	et_model = custom_model if et_model == "Other" else et_model
	lp_model = custom_model if lp_model == "Other" else lp_model
	ea_model = custom_embedding_model if ea_model == "Other" else ea_model

	# Run pipeline with progress tracking
	result = run_pipeline(text, source_url, ie_model, et_model, ea_model, lp_model, similarity_threshold, progress)
	if result.startswith("Error:"):
		return (
			result,
			None,
			get_metrics_box(),
		)
	try:
		# Create visualization without progress tracking
		result_dict = json.loads(result)
		graph_url, _ = create_graph_visualization(result_dict)
		graph_html_content = f"""
        <div style="text-align: center; padding: 10px; margin-top: -20px;">
            <h2 style="margin-bottom: 0.5em;">Entity Relationship Graph</h2>
            <em>Drag nodes • Scroll to zoom • Drag background to pan</em>
        </div>
        <div id="iframe-container"">
            <iframe src="{graph_url}"
            width="100%"
            height="700"
            frameborder="0"
            scrolling="no"
            style="display: block; clip-path: inset(13px 3px 5px 3px); overflow: hidden;">
            </iframe>
        </div>
        <div style="text-align: center; ">
            <a href="{graph_url}" target="_blank" style="color: #7c4dff; text-decoration: none;">
            🚀 Open in New Tab
            </a>
        </div>"""

		ie_metrics = f"Model: {ie_model}<br>Time: {result_dict['IE']['response_time']:.2f}s<br>Cost: ${result_dict['IE']['model_usage']['total']['cost']:.6f}"
		et_metrics = f"Model: {et_model}<br>Time: {result_dict['ET']['response_time']:.2f}s<br>Cost: ${result_dict['ET']['model_usage']['total']['cost']:.6f}"
		ea_metrics = f"Model: {ea_model}<br>Time: {result_dict['EA']['response_time']:.2f}s<br>Cost: ${result_dict['EA']['model_usage']['total']['cost']:.6f}"
		lp_metrics = f"Model: {lp_model}<br>Time: {result_dict['LP']['response_time']:.2f}s<br>Cost: ${result_dict['LP']['model_usage']['total']['cost']:.6f}"

		metrics_table = get_metrics_box(ie_metrics, et_metrics, ea_metrics, lp_metrics)

		return result, graph_html_content, metrics_table
	except Exception as e:
		import traceback

		traceback.print_exc()
		return (
			result,
			f"<div style='color: red; text-align: center; padding: 20px;'>Error creating graph: {str(e)}</div>",
			get_metrics_box(),
		)


def clear_outputs():
	"""Clear all outputs when run button is clicked"""
	return "", None, get_metrics_box()


def is_valid_source_url(source_url: str) -> bool:
	"""Basic URL validation for Gradio input."""
	if not source_url or not isinstance(source_url, str):
		return False
	candidate = source_url.strip()
	if "://" not in candidate:
		candidate = f"https://{candidate}"
	parsed = urlparse(candidate)
	return parsed.scheme in {"http", "https"} and bool(parsed.netloc and " " not in parsed.netloc)


def build_interface(warning: str = None):
	with gr.Blocks(title="CTINexus") as ctinexus:
		gr.HTML("""
            <style>
                .image-container {
                    background: none !important;
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 auto !important;
                    display: flex !important;
                    justify-content: center !important;
                }
                .image-container img {
                    border: none !important;
                    box-shadow: none !important;
                }

                .metric-label h2.output-class {
                    font-size: 0.9em !important;
                    font-weight: normal !important;
                    padding: 4px 8px !important;
                    line-height: 1.2 !important;
                }

                .metric-label th, td {
                    border: 1px solid var(--block-border-color) !important;
                }

                .metric-label .wrap {
                    display: none !important;
                }

                .note-text {
                    text-align: center !important;
                }

                .shadowbox {
                    background: var(--input-background-fill); !important;
                    border: 1px solid var(--block-border-color) !important;
                    border-radius: 4px !important;
                    padding: 8px !important;
                    margin: 4px 0 !important;
                }

                #resizable-results {
                    resize: both;
                    overflow: auto;
                    min-height: 200px;
                    min-width: 300px;
                    max-width: 100%;
                }

            </style>
        """)

		gr.Image(
			value=resolve_path("static", "logo.png"),
			width=100,
			height=100,
			show_label=False,
			elem_classes="image-container",
			interactive=False,
			show_download_button=False,
			show_fullscreen_button=False,
			show_share_button=False,
		)

		if warning:
			gr.Markdown(warning)

		with gr.Row():
			with gr.Column():
				input_source_selector = gr.Radio(
					choices=["CTI Report URL", "CTI Text"],
					value="CTI Report URL",
					label="Input Source",
				)
				url_input = gr.Textbox(
					label="CTI Report URL",
					placeholder="https://example.com/report",
					lines=1,
					visible=True,
				)
				text_input = gr.Textbox(
					label="Input Threat Intelligence",
					placeholder="Enter text for processing...",
					lines=10,
					visible=False,
				)
				gr.Markdown(
					"**Note:** Intelligence Extraction does best with a reasoning or full gpt model (e.g. o4-mini, gpt-4.1), Entity Tagging tends to need a mid level gpt model (gpt-4o-mini, gpt-4.1-mini).",
					elem_classes=["note-text"],
				)

				def toggle_input_source(source_choice):
					use_text_input = source_choice == "CTI Text"
					return gr.update(visible=use_text_input), gr.update(visible=not use_text_input)

				input_source_selector.change(
					fn=toggle_input_source,
					inputs=[input_source_selector],
					outputs=[text_input, url_input],
				)

				with gr.Row():
					with gr.Column(scale=1):
						provider_dropdown = gr.Dropdown(
							choices=list(MODELS.keys()) if MODELS else [],
							label="AI Provider",
							value="OpenAI" if "OpenAI" in MODELS else (list(MODELS.keys())[0] if MODELS else None),
						)
					with gr.Column(scale=2):
						ie_dropdown = gr.Dropdown(
							choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")]
							if provider_dropdown.value
							else [],
							label="Intelligence Extraction Model",
							value=get_model_choices(provider_dropdown.value)[0][1]
							if provider_dropdown.value and get_model_choices(provider_dropdown.value)
							else None,
						)

					with gr.Column(scale=2):
						et_dropdown = gr.Dropdown(
							choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")]
							if provider_dropdown.value
							else [],
							label="Entity Tagging Model",
							value=get_model_choices(provider_dropdown.value)[0][1]
							if provider_dropdown.value and get_model_choices(provider_dropdown.value)
							else None,
						)
				with gr.Row():
					with gr.Column(scale=2):
						ea_dropdown = gr.Dropdown(
							choices=get_embedding_model_choices(provider_dropdown.value) + [("Other", "Other")]
							if provider_dropdown.value
							else [],
							label="Entity Alignment Model",
							value=get_embedding_model_choices(provider_dropdown.value)[0][1]
							if provider_dropdown.value and get_embedding_model_choices(provider_dropdown.value)
							else None,
						)
					with gr.Column(scale=1):
						similarity_slider = gr.Slider(
							minimum=0.0,
							maximum=1.0,
							value=0.6,
							step=0.05,
							label="Alignment Threshold (higher = more strict)",
						)
					with gr.Column(scale=2):
						lp_dropdown = gr.Dropdown(
							choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")]
							if provider_dropdown.value
							else [],
							label="Link Prediction Model",
							value=get_model_choices(provider_dropdown.value)[0][1]
							if provider_dropdown.value and get_model_choices(provider_dropdown.value)
							else None,
						)

				# Custom model input fields
				with gr.Row():
					with gr.Column(scale=1):
						custom_model_input = gr.Textbox(
							label="Custom Model (if 'Other' is selected)",
							placeholder="Enter custom model name...",
							visible=False,
						)
					with gr.Column(scale=1):
						custom_embedding_model_input = gr.Textbox(
							label="Custom Embedding Model (if 'Other' is selected)",
							placeholder="Enter custom embedding model name...",
							visible=False,
						)

				def toggle_custom_model_inputs(ie_value, et_value, ea_value, lp_value):
					show_custom_model = any(value == "Other" for value in [ie_value, et_value, lp_value])
					show_custom_embedding_model = ea_value == "Other"
					return gr.update(visible=show_custom_model), gr.update(visible=show_custom_embedding_model)

				ie_dropdown.change(
					fn=toggle_custom_model_inputs,
					inputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
					outputs=[custom_model_input, custom_embedding_model_input],
				)

				et_dropdown.change(
					fn=toggle_custom_model_inputs,
					inputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
					outputs=[custom_model_input, custom_embedding_model_input],
				)

				ea_dropdown.change(
					fn=toggle_custom_model_inputs,
					inputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
					outputs=[custom_model_input, custom_embedding_model_input],
				)

				lp_dropdown.change(
					fn=toggle_custom_model_inputs,
					inputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
					outputs=[custom_model_input, custom_embedding_model_input],
				)

				run_all_button = gr.Button("Run", variant="primary")
		with gr.Row():
			metrics_table = gr.Markdown(
				value=get_metrics_box(),
				elem_classes=["metric-label"],
			)

		with gr.Row():
			with gr.Column(scale=1):
				results_box = gr.Code(
					label="Results",
					language="json",
					interactive=False,
					show_line_numbers=False,
					elem_classes=["results-box"],
					elem_id="resizable-results",
				)
			with gr.Column(scale=2):
				graph_output = gr.HTML(
					label="Entity Relationship Graph",
					value="""
                        <div style="text-align: center; margin-top: -20px;">
                            <h2 style="margin-bottom: 0.5em;">Entity Relationship Graph</h2>
                            <em>No graph to display yet. Click "Run" to generate a visualization.</em>
                        </div>
                    """,
				)

		def update_model_choices(
			provider,
		) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
			model_choices = get_model_choices(provider) + [("Other", "Other")]
			embedding_choices = get_embedding_model_choices(provider) + [("Other", "Other")]

			# Create dropdowns with updated choices and default values
			ie_dropdown_update = gr.Dropdown(
				choices=model_choices, value=model_choices[0][1] if model_choices else None
			)
			et_dropdown_update = gr.Dropdown(
				choices=model_choices, value=model_choices[0][1] if model_choices else None
			)
			ea_dropdown_update = gr.Dropdown(
				choices=embedding_choices, value=embedding_choices[0][1] if embedding_choices else None
			)
			lp_dropdown_update = gr.Dropdown(
				choices=model_choices, value=model_choices[0][1] if model_choices else None
			)

			return (
				ie_dropdown_update,
				et_dropdown_update,
				ea_dropdown_update,
				lp_dropdown_update,
			)

		# Connect buttons to their respective functions
		provider_dropdown.change(
			fn=update_model_choices,
			inputs=[provider_dropdown],
			outputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
		)

		run_all_button.click(
			fn=clear_outputs,
			inputs=[],
			outputs=[results_box, graph_output, metrics_table],
		).then(
			fn=process_and_visualize,
			inputs=[
				input_source_selector,
				text_input,
				url_input,
				ie_dropdown,
				et_dropdown,
				ea_dropdown,
				lp_dropdown,
				similarity_slider,
				provider_dropdown,
				custom_model_input,
				custom_embedding_model_input,
			],
			outputs=[results_box, graph_output, metrics_table],
		)

	ctinexus.launch()
