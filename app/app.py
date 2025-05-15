import json
import os
import sys
import traceback

import gradio as gr
from cti_processor import PostProcessor, preprocessor
from dotenv import load_dotenv
from graph_constructor import Linker, Merger
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


def get_model_choices(provider):
    """Get model choices with descriptions for the dropdown"""
    return [(desc, key) for key, desc in MODELS[provider].items()]


def main():
    # Accept "text as an argument via command line
    warning = None
    if not check_api_key():
        warning = "⚠️ **Warning: OPENAI_API_KEY environment variable is not set in the .env file. The application will not function correctly.**"

    if len(sys.argv) > 1:
        if warning:
            print(warning)
        text = sys.argv[1]
        linking_result = run_pipeline(text)
        print(linking_result)
    else:
        # Create the Gradio interface
        with gr.Blocks(title="CTINexus") as ctinexus:
            gr.Markdown("# CTINexus")
            gr.Markdown("Run the CTI analysis steps.")

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
                        provider_dropdown = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            label="AI Provider",
                            value="OpenAI"
                            if "OpenAI" in MODELS
                            else list(MODELS.keys())[0],
                        )
                        model_dropdown = gr.Dropdown(
                            choices=get_model_choices(provider_dropdown.value),
                            label="Model",
                            value=get_model_choices(provider_dropdown.value)[0][1],
                        )
                    run_all_button = gr.Button("Run", variant="primary")

            with gr.Row():
                with gr.Column():
                    results_box = gr.Textbox(
                        label="Results",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes=["results-box"],
                    )

            # Add custom CSS for the results box
            gr.HTML("""
                <style>
                    .results-box {
                        overflow-y: auto !important;
                        max-height: 600px !important;
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
                fn=run_pipeline,
                inputs=[text_input, model_dropdown],
                outputs=[results_box],
            )

        ctinexus.launch()


if __name__ == "__main__":
    main()
