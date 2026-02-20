<div align="center">
  <img src="https://raw.githubusercontent.com/peng-gao-lab/CTINexus/main/ctinexus/static/logo.png" alt="Logo" width="200">
  <h1 align="center">Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models</h1>
</div>

<p align="center">
  <a href='https://arxiv.org/abs/2410.21060'><img src='https://img.shields.io/badge/Paper-Arxiv-crimson'></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lavender.svg" alt="License: MIT"></a>
  <a href='https://ctinexus.github.io/' target='_blank'><img src='https://img.shields.io/badge/Project-Website-turquoise'></a>
  <a href="https://pepy.tech/projects/ctinexus" target='_blank'><img src="https://static.pepy.tech/personalized-badge/ctinexus?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BRIGHTGREEN&left_text=Downloads" alt="PyPI Downloads"></a>
</p>

---

## News & Updates

📦 [2025/10] CTINexus Python package released! Install with `pip install ctinexus` for seamless integration into your Python projects.

🌟 [2025/07] CTINexus now features an intuitive Gradio interface! Submit threat intelligence text and instantly visualize extracted interactive graphs.

🔥 [2025/04] We released the camera-ready paper on [arxiv](https://arxiv.org/pdf/2410.21060).

🔥 [2025/02] CTINexus is accepted at 2025 IEEE European Symposium on Security and Privacy ([Euro S&P](https://eurosp2025.ieee-security.org/index.html)).


## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported AI Providers](#supported-ai-providers)
- [Getting Started](#getting-started)
  - [Option 1: Python Package](#python-package)
  - [Option 2: Web Interface (Local)](#web-interface)
  - [Option 3: Docker](#docker-setup)
- [Command Line Interface](#command-line)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

**CTINexus** is a framework that leverages optimized in-context learning (ICL) of large language models (LLMs) to automatically extract cyber threat intelligence (CTI) from unstructured text and construct cybersecurity knowledge graphs (CSKG).

<p align="center">
  <img src="https://raw.githubusercontent.com/peng-gao-lab/CTINexus/main/ctinexus/static/overview.png" alt="CTINexus Framework Overview" width="600"/>
</p>

The framework processes threat intelligence reports to:

- 🔍 **Extract cybersecurity entities** (malware, vulnerabilities, tactics, IOCs)
- 🔗 **Identify relationships** between security concepts
- 📊 **Construct knowledge graphs** with interactive visualizations
- ⚡ **Require minimal configuration** - no extensive training data or parameter tuning needed

---

## Features

### Core Pipeline Components

1. **Intelligence Extraction (IE)**
   - Automatically extracts cybersecurity entities and relationships from unstructured text
   - Uses optimized prompt construction and demonstration retrieval

2. **Hierarchical Entity Alignment**
   - **Entity Typing (ET)**: Classifies entities by semantic type
   - **Entity Merging (EM)**: Canonicalizes entities and removes redundancy with IOC protection

3. **Link Prediction (LP)**
   - Predicts and adds missing relationships to complete the knowledge graph

4. **Interactive Visualization**
   - Network graph visualization of the constructed cybersecurity knowledge graph

<p align="center">
  <img src="https://raw.githubusercontent.com/peng-gao-lab/CTINexus/main/ctinexus/static/webui.png" alt="CTINexus WebUI" width="600"/>
</p>

---

## Supported AI Providers

CTINexus supports multiple AI providers for flexibility:

| Provider | Models | Setup Required |
|----------|--------|----------------|
| **OpenAI** | GPT-4, GPT-4o, o1, o3, etc. | API Key |
| **Google Gemini** | Gemini 2.0, 2.5 Flash, etc. | API Key |
| **AWS Bedrock** | Claude, Nova, Llama, DeepSeek, etc. | AWS Credentials |
| **Ollama** | Llama, Mistral, Qwen, Gemma, etc. | Local Installation (FREE) |

> Note: When using Ollama models, use the **📖 [Ollama Setup Guide](docs/ollama-guide.md)**.

---

## Getting Started


<a id="python-package"></a>

### 📦 Option 1: Python Package

#### Installation

```bash
pip install ctinexus
```


#### Configuration

Create a `.env` file in your project directory with credentials for at least one provider. Look at [.env.example](.env.example) for reference.

#### Usage

```python
from ctinexus import process_cti_report
from dotenv import load_dotenv

# Load API credentials
load_dotenv()

# Process threat intelligence text
text = """
APT29 used PowerShell to download additional malware from command-and-control
server at 192.168.1.100. The attack exploited CVE-2023-1234 in Microsoft Exchange.
"""

result = process_cti_report(
    text=text,
    provider="openai",  # optional: auto-detected if not specified
    model="gpt-4",      # optional: uses default if not specified
    similarity_threshold=0.6,
    output="results.json"  # optional: save results to file
)

# Access results
print(f"Graph saved to: {result['entity_relation_graph']}")
# Open the HTML file in your browser to view the interactive graph

# Or process from a CTI report/blog URL
result = process_cti_report(
    source_url="https://example.com/threat-report",
    provider="openai",
    model="gpt-4",
)
```

**API Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | None | Threat intelligence text to process (required if `source_url` is not provided) |
| `source_url` | str | None | CTI report/blog URL to ingest and process (required if `text` is not provided) |
| `provider` | str | Auto-detect | `"openai"`, `"gemini"`, `"aws"`, or `"ollama"` |
| `model` | str | Provider default | Model name (e.g., `"gpt-4o"`, `"gemini-2.0-flash"`) |
| `embedding_model` | str | Provider default | Embedding model for entity alignment |
| `similarity_threshold` | float | 0.6 | Entity similarity threshold (0.0-1.0) |
| `output` | str | None | Path to save JSON results |

**Note**: `text` and `source_url` are mutually exclusive. Provide exactly one input source.

**Return Value:**

The function returns a dictionary with complete analysis results:

```python
{
    "text": "Original input text",
    "IE": {"triplets": [...]},  # Extracted entities and relationships
    "ET": {"typed_triplets": [...]},  # Entities with type classifications
    "EA": {"aligned_triplets": [...]},  # Canonicalized entities
    "LP": {"predicted_links": [...]},  # Predicted relationships
    "entity_relation_graph": "path/to/graph.html"  # Interactive visualization
}
```

---

<a id="web-interface"></a>

### 🖥️ Option 2: Web Interface (Local Setup)

#### Installation

```bash
git clone https://github.com/peng-gao-lab/CTINexus.git
cd CTINexus

# Create and activate virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
# .venv\Scripts\activate

# Install the package
pip install -e .
```


#### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
```

#### Usage

**1. Launch the application:**

```bash
ctinexus
```

**2. Access the web interface:**

Open your browser to: **http://127.0.0.1:7860**

**3. Process threat intelligence:**

1. **Paste** threat intelligence text into the input area
2. **Select** your AI provider and model from dropdowns
3. **Click** "Run" to analyze
4. **View** extracted entities, relationships, and interactive graph
5. **Export** results as JSON or save graph images

---

<a id="docker-setup"></a>

### 🐳 Option 3: Docker (Containerized Setup)

**Prerequisites:**
- Install [Docker Desktop](https://docs.docker.com/get-docker/)

**Setup:**

```bash
# Clone the repository
git clone https://github.com/peng-gao-lab/CTINexus.git
cd CTINexus

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
```


#### Usage

**1. Build and start:**

```bash
# Run in foreground
docker compose up --build

# OR run in background (detached mode)
docker compose up -d --build

# View logs (if running in background)
docker compose logs -f
```

**2. Access the application:**

Open your browser to: **http://localhost:8000**

**3. Process threat intelligence:**

1. **Paste** threat intelligence text into the input area
2. **Select** your AI provider and model from dropdowns
3. **Click** "Run" to analyze
4. **View** extracted entities, relationships, and interactive graph
5. **Export** results as JSON or save graph images

---

<a id="command-line"></a>

## ⚡ Command Line Interface

The CLI works with **any installation method** and is perfect for automation and batch processing.

### Basic Usage

```bash
# Process a file
ctinexus --input-file report.txt

# Process text directly
ctinexus --text "APT29 exploited CVE-2023-1234 using PowerShell..."

# Specify provider and model
ctinexus -i report.txt --provider openai --model gpt-4o

# Save to custom location
ctinexus -i report.txt --output results/analysis.json
```

**📖 [Complete CLI Documentation](docs/cli-guide.md)** - Detailed examples and all available options.

---

## Contributing

We warmly welcome contributions from the community! Whether you're interested in:

- 🐛 Fix bugs or add features
- 📖 Improve documentation
- 🎨 Enhance the UI/UX
- 🧪 Add tests or examples

Please check out our **[Contributing Guide](CONTRIBUTING.md)** for detailed information on how to get started, development setup, and submission guidelines.

## Citation

If you use CTINexus in your research, please cite our paper:

```bibtex
@inproceedings{cheng2025ctinexusautomaticcyberthreat,
      title={CTINexus: Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models},
      author={Yutong Cheng and Osama Bajaber and Saimon Amanuel Tsegai and Dawn Song and Peng Gao},
      booktitle={2025 IEEE European Symposium on Security and Privacy (EuroS\&P)},
      year={2025},
      organization={IEEE}
}
```

## License

The source code is licensed under the [MIT](LICENSE.txt) License.
We warmly welcome industry collaboration. If you’re interested in building on CTINexus or exploring joint initiatives, please email yutongcheng@vt.edu or saimon.tsegai@vt.edu, we’d be happy to set up a brief call to discuss ideas.
