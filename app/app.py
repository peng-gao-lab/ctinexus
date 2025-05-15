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


def check_api_key() -> bool:
    """Define Models and check if API KEYS are set"""
    if os.getenv("OPENAI_API_KEY"):
        MODELS["OpenAI"] = {
            "gpt-4.1": "GPT-4.1 — Flagship GPT model for complex tasks ($2 • $8)",
            "gpt-4o": "GPT-4o — Fast, intelligent, flexible GPT model ($2.5 • $10)",
            "gpt-4.1-mini": "GPT-4.1 Mini — Balanced for intelligence, speed, and cost ($0.4 • $1.6)",
            "gpt-4o-mini": "GPT-4o Mini — Fast, affordable small model for focused tasks ($0.15 • $0.6)",
            "gpt-4.1-nano": "GPT-4.1 Nano — Fastest, most cost-effective GPT-4.1 model ($0.1 • $0.4)",
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
    return True if MODELS else False


def run_extraction(config: DictConfig, text: str = None) -> dict:
    """Wrapper for Information Extraction"""
    return LLMExtractor(config).call(text)


def run_entity_typing(config: DictConfig, result: dict) -> dict:
    """Wrapper for Entity Typing"""
    return LLMTagger(config).call(result)


def run_entity_merging(config: DictConfig, result: dict) -> dict:
    """Wrapper for Entity Merging"""
    preprocessed_result = preprocessor(result)
    merged_result = Merger(config).call(preprocessed_result)
    return PostProcessor(config).call(merged_result)


def run_link_prediction(config: DictConfig, result) -> dict:
    """Wrapper for Link Prediction"""

    if not isinstance(result, dict):
        result = {"subgraphs": result}

    return Linker(config).call(result)


def run_pipeline(
    text: str = None, model: str = None, progress=gr.Progress(track_tqdm=False)
):
    """Run the entire pipeline in sequence"""
    if not text:
        return "Please enter some text to process."

    with initialize(version_base="1.2", config_path="config"):
        config = compose(
            config_name="config.yaml", overrides=[f"model={model}"] if model else []
        )

    try:
        progress(0, desc="Entity Extraction...")
        extraction_result = run_extraction(config, text)

        progress(0.3, desc="Entity Typing...")
        typing_result = run_entity_typing(config, extraction_result)

        progress(0.6, desc="Entity Merging...")
        merging_result = run_entity_merging(config, typing_result)

        progress(0.9, desc="Link Prediction...")
        linking_result = run_link_prediction(config, merging_result)

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
                    with gr.Column(scale=1):
                        provider_dropdown = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            label="AI Provider",
                            value="OpenAI"
                            if "OpenAI" in MODELS
                            else list(MODELS.keys())[0],
                        )
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value),
                            label="Model",
                            value=get_model_choices(provider_dropdown.value)[0][1],
                        )

                run_all_button = gr.Button("Run", variant="primary")

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

        # Add custom CSS for the results box
        gr.HTML("""
            <style>
                .results-box {
                    overflow-y: auto !important;
                    max-height: 600px !important;
                }
                .results-box .monaco-editor {
                    background-color: #27272a !important;
                }
            </style>
        """)

        def update_model_choices(provider):
            return gr.Dropdown(choices=get_model_choices(provider))

        # Connect buttons to their respective functions
        provider_dropdown.change(
            fn=update_model_choices,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        run_all_button.click(
            fn=process_and_visualize,
            inputs=[text_input, model_dropdown],
            outputs=[results_box, graph_output],
        )

    ctinexus.launch()


def get_model_choices(provider):
    """Get model choices with descriptions for the dropdown"""
    return [(desc, key) for key, desc in MODELS[provider].items()]


def process_and_visualize(text, model, progress=gr.Progress(track_tqdm=False)):
    # Run pipeline with progress tracking
    result = run_pipeline(text, model, progress)
    if result.startswith("Error:"):
        return result, None
    try:
        # Create visualization without progress tracking
        result_dict = json.loads(result)
        graph_fig = create_graph_visualization(result_dict)
        return result, graph_fig
    except Exception as e:
        return result, None


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
