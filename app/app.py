import json
import os
import sys

import gradio as gr
from cti_processor import PostProcessor, preprocessor
from dotenv import load_dotenv
from graph_constructor import Linker, Merger
from hydra import compose, initialize
from llm_processor import LLMExtractor, LLMTagger

load_dotenv()


def check_api_key():
    """Check if OPENAI_API_KEY is set"""
    return bool(os.getenv("OPENAI_API_KEY"))


def run_extraction(text: str = None) -> dict:
    """Wrapper for Information Extraction"""
    with initialize(version_base="1.2", config_path="config"):
        config = compose(config_name="ieConfig.yaml")

    return LLMExtractor(config).call(text)


def run_entity_typing(result: dict) -> dict:
    """Wrapper for Entity Typing"""
    with initialize(version_base="1.2", config_path="config"):
        config = compose(config_name="etConfig.yaml")

    return LLMTagger(config).call(result)


def run_entity_merging(result: dict) -> dict:
    """Wrapper for Entity Merging"""
    preprocessed_result = preprocessor(result)
    with initialize(version_base="1.2", config_path="config"):
        config = compose(config_name="emConfig.yaml")

    merged_result = Merger(config).call(preprocessed_result)
    with initialize(version_base="1.2", config_path="config"):
        config = compose(config_name="postConfig.yaml")

    return PostProcessor(config).call(merged_result)


def run_link_prediction(result: dict) -> dict:
    """Wrapper for Link Prediction"""
    with initialize(version_base="1.2", config_path="config"):
        config = compose(config_name="ltConfig.yaml")

    # Ensure result is properly formatted for the Linker
    if not isinstance(result, dict):
        result = {"subgraphs": result}

    return Linker(config).call(result)


def run_pipeline(text: str = None):
    """Run the entire pipeline in sequence"""
    extraction_result = run_extraction(text)
    typing_result = run_entity_typing(extraction_result)
    merging_result = run_entity_merging(typing_result)
    linking_result = run_link_prediction(merging_result)
    return json.dumps(linking_result, indent=4)


def main():
    # Accept "text as an argument via command line
    warning = None
    if not check_api_key():
        warning = "⚠️ **Warning: OPENAI_API_KEY environment variable is not set in the .env file. The application will not function correctly.**"

    if len(sys.argv) > 1:
        if warning:
            print(warning)

        text = sys.argv[1]

        extraction_result = run_extraction(text)
        typing_result = run_entity_typing(extraction_result)
        merging_result = run_entity_merging(typing_result)
        linking_result = run_link_prediction(merging_result)
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
                    run_all_button = gr.Button("Run", variant="primary")

            output = gr.Textbox(label="Result", lines=10)

            def run_extraction_with_text(text):
                if not text:
                    return {"error": "No text provided for Information Extraction"}
                return run_extraction(text)

            # Connect buttons to their respective functions
            run_all_button.click(fn=run_pipeline, inputs=text_input, outputs=output)

        ctinexus.launch()


if __name__ == "__main__":
    main()
