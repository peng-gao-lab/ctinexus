import argparse
import json
import os
import sys
import traceback

import gradio as gr
from cti_processor import PostProcessor, preprocessor
from dotenv import load_dotenv
from graph_constructor import Linker, Merger, create_graph_visualization
from hydra import compose, initialize
from llm_processor import LLMExtractor, LLMTagger
from omegaconf import DictConfig
from utils.http_server_utils import setup_http_server
from utils.path_utils import resolve_path

load_dotenv()

# Available models
MODELS = {}
EMBEDDING_MODELS = {}


def check_api_key() -> bool:
    """Define Models and check if API KEYS are set"""
    if os.getenv("OPENAI_API_KEY"):
        MODELS["OpenAI"] = {
            "o4-mini": "o4 Mini — Faster, more affordable reasoning model ($1.1 • $4.4)",
            "o3-mini": "o3 Mini — A small reasoning model alternative to o3 ($1.1 • $4.4)",
            "o3": "o3 — Most powerful reasoning model ($2 • $8)",
            "o3-pro": "o3 Pro — Version of o3 with more compute for better responses ($20 • $80)", 
            "gpt-4.1": "GPT-4.1 — Flagship GPT model for complex tasks ($2 • $8)",
            "gpt-4o": "GPT-4o — Fast, intelligent, flexible GPT model ($2.5 • $10)",
            "gpt-4": "GPT-4 — An older high-intelligence GPT model ($30 • $60)",
            "gpt-4-turbo": "GPT-4 Turbo — An older high-intelligence GPT model ($10 • $30)",
            "gpt-3.5-turbo": "GPT-3.5 Turbo — Legacy GPT model for cheaper chat and non-chat tasks ($0.5 • $1.5)",
            "gpt-4.1-mini": "GPT-4.1 Mini — Balanced for intelligence, speed, and cost ($0.4 • $1.6)",
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


def run_intel_extraction(config: DictConfig, text: str = None) -> dict:
    """Wrapper for Intelligence Extraction"""
    return LLMExtractor(config).call(text)


def run_entity_tagging(config: DictConfig, result: dict) -> dict:
    """Wrapper for Entity Tagging"""
    return LLMTagger(config).call(result)


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


def get_config(model: str = None, embedding_model: str = None, similarity_threshold: float = 0.6) -> DictConfig:
    provider = get_model_provider(model, embedding_model)
    model = model.split("/")[-1] if model else None
    embedding_model = embedding_model.split("/")[-1] if embedding_model else None
    
    with initialize(version_base="1.2", config_path="config"):
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
    ie_model: str = None,
    et_model: str = None,
    ea_model: str = None,
    lp_model: str = None,
    similarity_threshold: float = 0.6,
    progress=gr.Progress(track_tqdm=False),
):
    """Run the entire pipeline in sequence"""
    if not text:
        return "Please enter some text to process."

    try:
        config = get_config(ie_model, None, None)
        progress(0, desc="Intelligence Extraction...")
        extraction_result = run_intel_extraction(config, text)

        config = get_config(et_model, None, None)
        progress(0.3, desc="Entity Tagging...")
        tagging_result = run_entity_tagging(config, extraction_result)

        progress(0.6, desc="Entity Alignment...")
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
                text_input = gr.Textbox(
                    label="Input Threat Intelligence",
                    placeholder="Enter text for processing...",
                    lines=10,
                )
                gr.Markdown(
                    "**Note:** Intelligence Extraction does best with a reasoning or full gpt model (e.g. o4-mini, gpt-4.1), Entity Tagging tends to need a mid level gpt model (gpt-4o-mini, gpt-4.1-mini).",
                    elem_classes=["note-text"],
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        provider_dropdown = gr.Dropdown(
                            choices=list(MODELS.keys()) if MODELS else [],
                            label="AI Provider",
                            value="OpenAI"
                            if "OpenAI" in MODELS
                            else (list(MODELS.keys())[0] if MODELS else None),
                        )
                    with gr.Column(scale=2):
                        ie_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")] if provider_dropdown.value else [],
                            label="Intelligence Extraction Model",
                            value=get_model_choices(provider_dropdown.value)[0][1] if provider_dropdown.value and get_model_choices(provider_dropdown.value) else None,
                        )

                    with gr.Column(scale=2):
                        et_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")] if provider_dropdown.value else [],
                            label="Entity Tagging Model",
                            value=get_model_choices(provider_dropdown.value)[0][1] if provider_dropdown.value and get_model_choices(provider_dropdown.value) else None,
                        )
                with gr.Row():
                    
                    with gr.Column(scale=2):
                        ea_dropdown = gr.Dropdown(
                            choices=get_embedding_model_choices(
                                provider_dropdown.value
                            ) + [("Other", "Other")] if provider_dropdown.value else [],
                            label="Entity Alignment Model",
                            value=get_embedding_model_choices(provider_dropdown.value)[
                                0
                            ][1] if provider_dropdown.value and get_embedding_model_choices(provider_dropdown.value) else None,
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
                            choices=get_model_choices(provider_dropdown.value) + [("Other", "Other")] if provider_dropdown.value else [],
                            label="Link Prediction Model",
                            value=get_model_choices(provider_dropdown.value)[0][1] if provider_dropdown.value and get_model_choices(provider_dropdown.value) else None,
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
                choices=model_choices,
                value=model_choices[0][1] if model_choices else None
            )
            et_dropdown_update = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0][1] if model_choices else None
            )
            ea_dropdown_update = gr.Dropdown(
                choices=embedding_choices,
                value=embedding_choices[0][1] if embedding_choices else None
            )
            lp_dropdown_update = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0][1] if model_choices else None
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
            inputs=[text_input, ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown, similarity_slider, provider_dropdown, custom_model_input, custom_embedding_model_input],
            outputs=[results_box, graph_output, metrics_table],
        )

    ctinexus.launch()


def get_metrics_box(
    ie_metrics: str = "",
    et_metrics: str = "",
    ea_metrics: str = "",
    lp_metrics: str = "",
):
    """Generate metrics box HTML with optional metrics values"""
    return f'<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Intelligence Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Tagging</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Alignment</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td>{ie_metrics or ""}</td><td>{et_metrics or ""}</td><td>{ea_metrics or ""}</td><td>{lp_metrics or ""}</td></tr></table></div>'


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


def process_and_visualize(
    text, ie_model, et_model, ea_model, lp_model, similarity_threshold, provider_dropdown=None, custom_model_input=None, custom_embedding_model_input=None, progress=gr.Progress(track_tqdm=False)
):
    # Apply custom model only to dropdowns where 'Other' is selected
    custom_model = f"{provider_dropdown}/{custom_model_input}" if provider_dropdown else custom_model_input
    custom_embedding_model = f"{provider_dropdown}/{custom_embedding_model_input}" if provider_dropdown else custom_embedding_model_input

    ie_model = custom_model if ie_model == "Other" else ie_model
    et_model = custom_model if et_model == "Other" else et_model
    lp_model = custom_model if lp_model == "Other" else lp_model
    ea_model = custom_embedding_model if ea_model == "Other" else ea_model

    # Run pipeline with progress tracking
    result = run_pipeline(text, ie_model, et_model, ea_model, lp_model, similarity_threshold, progress)
    if result.startswith("Error:"):
        return (
            result,
            None,
            get_metrics_box(),
        )
    try:
        # Create visualization without progress tracking
        result_dict = json.loads(result)
        graph_url = create_graph_visualization(result_dict)
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
    except Exception:
        return (
            result,
            None,
            get_metrics_box(),
        )


def clear_outputs():
    """Clear all outputs when run button is clicked"""
    return "", None, get_metrics_box()


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="CTINexus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="Input threat intelligence text to process"
    )
    input_group.add_argument(
        "--input-file", "-i",
        type=str,
        help="Path to file containing threat intelligence text"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="AI provider to use: OpenAI, Gemini, AWS, or Ollama (auto-detected if not specified)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for all text processing steps (e.g., gpt-4o, o4-mini)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model for entity alignment (e.g., text-embedding-3-large)"
    )
    parser.add_argument(
        "--ie-model",
        type=str,
        help="Override model for Intelligence Extraction"
    )
    parser.add_argument(
        "--et-model", 
        type=str,
        help="Override model for Entity Tagging"
    )
    parser.add_argument(
        "--ea-model",
        type=str, 
        help="Override embedding model for Entity Alignment"
    )
    parser.add_argument(
        "--lp-model",
        type=str,
        help="Override model for Link Prediction"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for entity alignment (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (if not specified, saves to app/output/ directory)"
    )
    
    return parser


def get_default_models_for_provider(provider):
    defaults = {
        "OpenAI": {
            "model": "o4-mini",
            "embedding_model": "text-embedding-3-large"
        },
        "Gemini": {
            "model": "gemini-2.0-flash",
            "embedding_model": "gemini-embedding-001"
        },
        "AWS": {
            "model": "anthropic.claude-3-5-sonnet",
            "embedding_model": "amazon.titan-embed-text-v2:0"
        },
        "Ollama": {
            "model": "llama3.1:8b",
            "embedding_model": "nomic-embed-text"
        }
    }
    return defaults.get(provider, {})


def run_cmd_pipeline(args):
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        text = args.text
    
    if not text:
        print("Error: No input text provided")
        sys.exit(1)
    
    provider = args.provider
    available_providers = list(MODELS.keys())

    if provider:
        provider_matched = next((p for p in available_providers if provider.lower() == p.lower()), None)
        if not provider_matched:
            print(f"Error: Provider '{provider}' not available. Available providers: {available_providers}")
            sys.exit(1)
        provider = provider_matched
    else:
        # Auto-detect based on available API keys
        if available_providers:
            provider = available_providers[0]
        else:
            print("Error: No API keys configured")
            sys.exit(1)
    
    defaults = get_default_models_for_provider(provider)

    # Set models with fallbacks to defaults
    base_model = args.model or defaults.get("model")
    base_embedding_model = args.embedding_model or defaults.get("embedding_model")
    
    ie_model = f"{provider}/{args.ie_model or base_model}"
    et_model = f"{provider}/{args.et_model or base_model}"
    ea_model = f"{provider}/{args.ea_model or base_embedding_model}"
    lp_model = f"{provider}/{args.lp_model or base_model}"

    print(f"Running CTINexus with {provider} provider...")
    print(f"IE: {ie_model}, ET: {et_model}, EA: {ea_model}, LP: {lp_model}")
    
    try:
        result = run_pipeline(
            text=text,
            ie_model=ie_model,
            et_model=et_model, 
            ea_model=ea_model,
            lp_model=lp_model,
            similarity_threshold=args.similarity_threshold
        )
        
        if result.startswith("Error:"):
            print(result)
            sys.exit(1)

        # Determine output file
        if args.output:
            output_file = args.output
        elif args.input_file:
            # Use input filename with _output.json
            input_basename = os.path.basename(args.input_file)
            base_name = os.path.splitext(input_basename)[0]
            output_file = resolve_path("output", f"{base_name}_output.json")
        else:
            output_file = resolve_path("output", "output.json")
        
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Results written to: {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            print(result)
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    api_keys_available = check_api_key()

    run_gui = not args.text and not args.input_file
    
    if run_gui:
        # GUI mode
        warning = None
        if not api_keys_available:
            warning = "⚠️   Warning: No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.\n"
            print(warning)
        build_interface(warning)
    else:
        # Command line mode
        if not api_keys_available:
            print("⚠️   Warning: No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.\n")
            sys.exit(1)
        
        run_cmd_pipeline(args)


if __name__ == "__main__":
    # HTTP server to serve pyvis files
    setup_http_server()

    main()
