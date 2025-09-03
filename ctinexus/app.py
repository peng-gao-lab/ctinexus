# flake8: noqa

import argparse
import sys
import os
import json
import traceback
import logging
from dotenv import load_dotenv
import importlib.metadata

# Ensure the parent directory is in sys.path for absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ctinexus.graph_constructor import create_graph_visualization
from ctinexus.utils.gradio_utils import build_interface, run_pipeline
from ctinexus.utils.http_server_utils import setup_http_server
from ctinexus.utils.model_utils import (
    MODELS,
    check_api_key,
)

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="CTINexus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"CTINexus {importlib.metadata.version('ctinexus')}"
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
        help="Output file path (if not specified, saves to ctinexus/output/ directory)"
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
            logger.error(f"Input file '{args.input_file}' not found")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        text = args.text
    
    if not text:
        logger.error("No input text provided")
        sys.exit(1)
    
    provider = args.provider
    available_providers = list(MODELS.keys())

    if provider:
        provider_matched = next((p for p in available_providers if provider.lower() == p.lower()), None)
        if not provider_matched:
            logger.error(f"Provider '{provider}' not available. Available providers: {available_providers}")
            sys.exit(1)
        provider = provider_matched
    else:
        # Auto-detect based on available API keys
        if available_providers:
            provider = available_providers[0]
        else:
            logger.error("No API keys configured")
            sys.exit(1)
    
    defaults = get_default_models_for_provider(provider)

    # Set models with fallbacks to defaults
    base_model = args.model or defaults.get("model")
    base_embedding_model = args.embedding_model or defaults.get("embedding_model")
    
    ie_model = f"{provider}/{args.ie_model or base_model}"
    et_model = f"{provider}/{args.et_model or base_model}"
    ea_model = f"{provider}/{args.ea_model or base_embedding_model}"
    lp_model = f"{provider}/{args.lp_model or base_model}"

    logger.info(f"Running CTINexus with {provider} provider...")
    logger.info(f"IE: {ie_model}, ET: {et_model}, EA: {ea_model}, LP: {lp_model}")
    
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
            logger.error(result)
            sys.exit(1)

        # Determine output file
        if args.output:
            output_file = args.output
        elif args.input_file:
            # Use input filename with _output.json
            input_basename = os.path.basename(args.input_file)
            base_name = os.path.splitext(input_basename)[0]
            output_dir = os.path.join(os.getcwd(), "ctinexus_output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}_output.json")
        else:
            output_dir = os.path.join(os.getcwd(), "ctinexus_output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "output.json")

        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"Results written to: {output_file}")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
            logger.error(result)
            sys.exit(1)

        # Create Entity Relation Graph
        result_dict = json.loads(result)
        _, filepath = create_graph_visualization(result_dict)
        logger.info(f"Entity Relation Graph: {filepath}")

        
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    setup_logging()
    
    # HTTP server to serve pyvis files
    setup_http_server()

    parser = create_argument_parser()
    args = parser.parse_args()

    api_keys_available = check_api_key()

    run_gui = not args.text and not args.input_file
    
    if run_gui:
        # GUI mode
        warning = None
        if not api_keys_available:
            warning = "⚠️   Warning: No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.\n"
            logger.warning(warning.strip())
        build_interface(warning)
    else:
        # Command line mode
        if not api_keys_available:
            logger.warning("No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.")
            sys.exit(1)
        
        run_cmd_pipeline(args)


if __name__ == "__main__":

    main()
