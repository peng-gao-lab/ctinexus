# Using CTI Nexus with Ollama

CTI Nexus supports local LLM inference through Ollama, providing a free and private alternative to cloud-based AI providers.

## Prerequisites

Install Ollama from [ollama.ai](https://ollama.ai/download)

## Setup

### 1. Start Ollama Service

```bash
# Start Ollama (runs on http://localhost:11434 by default)
ollama serve
```

### 2. Pull Models

You need at least one text generation and one embedding model. You can use any Ollama model.

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Configure Environment

In your `.env` file, set the Ollama base URL. When using the Python local installation, set it to `http://localhost:11434`. When using Docker, set it to `http://host.docker.internal:11434`. In both cases, this assumes default Ollama port of `11434`. If your Ollama is running at a different port, use that port.

```bash
# Ollama Configuration
# Set to the default (http://localhost:11434).
# For docker, set to http://host.docker.internal:11434
OLLAMA_BASE_URL=
```

## Usage

### Command Line

```bash
# Use default Ollama models
ctinexus -i report.txt --provider Ollama

# Specify custom models
ctinexus -i report.txt --provider Ollama --model llama3.1:70b --embedding-model nomic-embed-text
```

### Web Interface

1. Start the web interface: `ctinexus`
1. Select "Ollama" as provider in the dropdown
1. Choose your preferred models
1. Process CTI text
