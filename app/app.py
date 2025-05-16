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

load_dotenv()

# Available models
MODELS = {}
EMBEDDING_MODELS = {}


def check_api_key() -> bool:
    """Define Models and check if API KEYS are set"""
    if os.getenv("OPENAI_API_KEY"):
        MODELS["OpenAI"] = {
            "o4-mini": "o4 Mini — Faster, more affordable reasoning model ($1.1 • $4.4)",
            "o3-mini": "o3 Mini — A small model alternative to o3 ($1.1 • $4.4)",
            "o3": "o3 — Most powerful reasoning model ($10 • $40)",
            "gpt-4.1": "GPT-4.1 — Flagship GPT model for complex tasks ($2 • $8)",
            "gpt-4o": "GPT-4o — Fast, intelligent, flexible GPT model ($2.5 • $10)",
            "gpt-4.1-mini": "GPT-4.1 Mini — Balanced for intelligence, speed, and cost ($0.4 • $1.6)",
            "gpt-4o-mini": "GPT-4o Mini — Fast, affordable small model for focused tasks ($0.15 • $0.6)",
            "gpt-4.1-nano": "GPT-4.1 Nano — Fastest, most cost-effective GPT-4.1 model ($0.1 • $0.4)",
        }
        EMBEDDING_MODELS["OpenAI"] = {
            "text-embedding-3-large": "Text Embedding 3 Large — Most capable embedding model ($0.13)",
            "text-embedding-3-small": "Text Embedding 3 Small — Small embedding model ($0.02)",
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
            "amazon.titan-embed-text-v2:0": "Titan Embed Text 2 — Balanced for intelligence and efficiency in text ($0.12)",
        }
    return True if MODELS else False


def run_extraction(config: DictConfig, text: str = None) -> dict:
    """Wrapper for Information Extraction"""
    return LLMExtractor(config).call(text)


def run_entity_typing(config: DictConfig, result: dict) -> dict:
    """Wrapper for Entity Typing"""
    return LLMTagger(config).call(result)


def run_entity_aggregation(config: DictConfig, result: dict) -> dict:
    """Wrapper for Entity Aggregation"""
    preprocessed_result = preprocessor(result)
    merged_result = Merger(config).call(preprocessed_result)
    return PostProcessor(config).call(merged_result)


def run_link_prediction(config: DictConfig, result) -> dict:
    """Wrapper for Link Prediction"""

    if not isinstance(result, dict):
        result = {"subgraphs": result}

    return Linker(config).call(result)


def get_model_provider(model):
    for provider, models in MODELS.items():
        if model in models:
            return provider
    return None


def get_config(model: str = None, embedding_model: str = None) -> DictConfig:
    provider = get_model_provider(model)
    with initialize(version_base="1.2", config_path="config"):
        overrides = []
        if model:
            overrides.append(f"model={model}")
        if embedding_model:
            overrides.append(f"embedding_model={embedding_model}")
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
    progress=gr.Progress(track_tqdm=False),
):
    """Run the entire pipeline in sequence"""
    if not text:
        return "Please enter some text to process."

    try:
        config = get_config(ie_model, None)
        progress(0, desc="Entity Extraction...")
        extraction_result = run_extraction(config, text)

        config = get_config(et_model, None)
        progress(0.3, desc="Entity Typing...")
        typing_result = run_entity_typing(config, extraction_result)

        progress(0.6, desc="Entity Aggregation...")
        config = get_config(None, ea_model)
        aggregation_result = run_entity_aggregation(config, typing_result)

        config = get_config(lp_model, None)
        progress(0.9, desc="Link Prediction...")
        linking_result = run_link_prediction(config, aggregation_result)

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

                .shadowbox {
                    background: #27272a !important;
                    border: 1px solid #444444 !important;
                    border-radius: 4px !important;
                    padding: 8px !important;
                    margin: 4px 0 !important;
                }
                
            </style>
        """)

        gr.Image(
            value="app/static/logo.png",
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
                with gr.Row():
                    with gr.Column():
                        provider_dropdown = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            label="AI Provider",
                            value="OpenAI"
                            if "OpenAI" in MODELS
                            else list(MODELS.keys())[0],
                        )
                    with gr.Column():
                        ie_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value),
                            label="Entity Extraction Model",
                            value=get_model_choices(provider_dropdown.value)[0][1],
                        )
                with gr.Row():
                    with gr.Column():
                        et_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value),
                            label="Entity Typing Model",
                            value=get_model_choices(provider_dropdown.value)[0][1],
                        )
                    with gr.Column():
                        ea_dropdown = gr.Dropdown(
                            choices=get_embedding_model_choices(provider_dropdown.value),
                            label="Entity Aggregation Model",
                            value=get_embedding_model_choices(provider_dropdown.value)[0][1],
                        )
                    with gr.Column():
                        lp_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value),
                            label="Link Prediction Model",
                            value=get_model_choices(provider_dropdown.value)[0][1],
                        )
                run_all_button = gr.Button("Run", variant="primary")
        with gr.Row():
            metrics_table = gr.Markdown(
                value='<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Information Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Typing</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Aggregation</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td></td><td></td><td></td><td></td></tr></table></div>',
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
                )
            with gr.Column(scale=2):
                graph_output = gr.Plot(
                    label="Entity Relationship Graph",
                    show_label=True,
                )

        def update_model_choices(
            provider,
        ) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown]:
            model_dropdown = gr.Dropdown(choices=get_model_choices(provider))
            embedded_model_dropdown = gr.Dropdown(choices=get_embedding_model_choices(provider))
            return model_dropdown, model_dropdown, embedded_model_dropdown, model_dropdown

        # Connect buttons to their respective functions
        provider_dropdown.change(
            fn=update_model_choices,
            inputs=[provider_dropdown],
            outputs=[ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
        )

        run_all_button.click(
            fn=process_and_visualize,
            inputs=[text_input, ie_dropdown, et_dropdown, ea_dropdown, lp_dropdown],
            outputs=[results_box, graph_output, metrics_table],
        )

    ctinexus.launch()


def get_model_choices(provider):
    """Get model choices with descriptions for the dropdown"""
    return [(desc, key) for key, desc in MODELS[provider].items()]

def get_embedding_model_choices(provider):
    """Get model choices with descriptions for the dropdown"""
    return [(desc, key) for key, desc in EMBEDDING_MODELS[provider].items()]


def process_and_visualize(
    text, ie_model, et_model, ea_model, lp_model, progress=gr.Progress(track_tqdm=False)
):
    # Run pipeline with progress tracking
    result = run_pipeline(text, ie_model, et_model, ea_model, lp_model, progress)
    if result.startswith("Error:"):
        return result, None, '<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Information Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Typing</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Aggregation</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td></td><td></td><td></td><td></td></tr></table></div>'
    try:
        # Create visualization without progress tracking
        result_dict = json.loads(result)
        graph_fig = create_graph_visualization(result_dict)

        # Extract metrics
        ie_metrics = f"Model: {ie_model}<br>Time: {result_dict['IE']['response_time']:.2f}s<br>Cost: ${result_dict['IE']['model_usage']['total']['cost']:.6f}"
        et_metrics = f"Model: {et_model}<br>Time: {result_dict['ET']['response_time']:.2f}s<br>Cost: ${result_dict['ET']['model_usage']['total']['cost']:.6f}"
        ea_metrics = f"Model: {ea_model}<br>Time: {result_dict['EA']['response_time']:.2f}s<br>Cost: ${result_dict['EA']['model_usage']['total']['cost']:.6f}"
        lp_metrics = f"Model: {lp_model}<br>Time: {result_dict['LP']['response_time']:.2f}s<br>Cost: ${result_dict['LP']['model_usage']['total']['cost']:.6f}"

        metrics_table = f'<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Information Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Typing</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Aggregation</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td>{ie_metrics}</td><td>{et_metrics}</td><td>{ea_metrics}</td><td>{lp_metrics}</td></tr></table></div>'

        return result, graph_fig, metrics_table
    except Exception as e:
        return result, None, '<div class="shadowbox"><table style="width: 100%; text-align: center; border-collapse: collapse;"><tr><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Information Extraction</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Typing</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Entity Aggregation</th><th style="width: 25%; border-bottom: 1px solid var(--block-border-color);">Link Prediction</th></tr><tr><td></td><td></td><td></td><td></td></tr></table></div>'


def main():
    # Accept "text as an argument via command line
    warning = None
    if not check_api_key():
        warning = "⚠️ **Warning: OPENAI_API_KEY environment variable is not set in the .env file. The application will not function correctly.**"
        print(warning)

    if len(sys.argv) > 1:
        result, _ = run_pipeline(sys.argv[1])
        print(result)
    else:
        # Create the Gradio interface
        build_interface(warning)


if __name__ == "__main__":
    main()
