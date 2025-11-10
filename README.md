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

**CTINexus** is a framework that leverages optimized in-context learning (ICL) of large language models (LLMs) for automatic cyber threat intelligence (CTI) knowledge extraction and cybersecurity knowledge graph (CSKG) construction.
CTINexus adapts to various cybersecurity ontologies with minimal annotated examples and provides a user-friendly web interface for instant threat intelligence analysis.

<p align="center">
  <img src="https://raw.githubusercontent.com/peng-gao-lab/CTINexus/main/ctinexus/static/overview.png" alt="CTINexus Framework Overview" width="500"/>
</p>

### What CTINexus Does

The framework automatically processes unstructured threat intelligence reports to:

- **Extract cybersecurity entities** (malware, vulnerabilities, tactics, IOCs)
- **Identify relationships** between security concepts
- **Construct knowledge graphs** with interactive visualizations
- **Require minimal configuration** - no extensive data or parameter tuning needed

### Core Components

- **Intelligence Extraction (IE)**: Automatically extracts cybersecurity entities and relationships from unstructured text using optimized prompt construction and demonstration retrieval
- **Hierarchical Entity Alignment**: Canonicalizes extracted knowledge and removes redundancy through:
  - **Entity Typing (ET)**: Groups mentions of the same semantic type
  - **Entity Merging (EM)**: Merges mentions referring to the same entity with IOC (Indicator of Compromise) protection
- **Link Prediction (LP)**: Predicts and adds missing relationships to complete the knowledge graph
- **Graph Visualization**: Interactive network visualization of the constructed cybersecurity knowledge graph

<p align="center">
  <img src="https://raw.githubusercontent.com/peng-gao-lab/CTINexus/main/ctinexus/static/webui.png" alt="CTINexus WebUI" width="500"/>
</p>

## News

📦 [2025/09/03] CTINexus Python package released! Install with `pip install ctinexus` for seamless integration into your Python projects.

🌟 [2025/07/29] CTINexus now features an intuitive Gradio interface! Submit threat intelligence text and instantly visualize extracted interactive graphs.

🔥 [2025/04/21] We released the camera-ready paper on [arxiv](https://arxiv.org/pdf/2410.21060).

🔥 [2025/02/12] CTINexus is accepted at 2025 IEEE European Symposium on Security and Privacy ([Euro S&P](https://eurosp2025.ieee-security.org/index.html)).

## Quick Start

You can use CTINexus in three ways:

- **📦 Python Package**: Python package for easy integration
- **⚡ Command Line**: For automation and batch processing → **[📖 CLI Guide](docs/cli-guide.md)**
- **🖥️ Web Interface**: User-friendly GUI for interactive analysis (follow the setup below)

### Supported Models

CTINexus supports the following AI providers:
**OpenAI**, **Gemini**, **AWS**, **Ollama**

All models from these providers are supported. If you would like to see additional providers integrated, please open a feature request issue [here](https://github.com/peng-gao-lab/CTINexus/issues).

<a id="python-package"></a>

## 📦 Using as a Python Package

CTINexus can be used as a Python library for seamless integration into your projects.

### Installation

```bash
pip install ctinexus
```

### Configuration

Before using CTINexus, you need to configure API keys. Create a `.env` file in your project directory with your credentials. Look at the [example env](.env.example) for reference.

### Usage

```python
from ctinexus import process_cti_report
from dotenv import load_dotenv

load_dotenv()

# Example usage
text = "Your CTI text here"

result = process_cti_report(
    text=text,
    provider="openai",  # optional: auto-detected if not specified
    model="gpt-4",      # optional: uses default if not specified
    similarity_threshold=0.6,
    output="results.json"  # optional: save results to file
)

# Access results
print(f"Graph:", result["entity_relation_graph"])
# Outputs the html file with the graph visualization.
# Open the html file on your browser to see the results.
```

### Parameters

- `text` (str): The threat intelligence report text to process
- `provider` (str, optional): AI provider ("openai", "gemini", "aws", "ollama"). Auto-detected from available keys if not specified
- `model` (str, optional): Specific model name (e.g., "gpt-4", "gemini-pro")
- `embedding_model` (str, optional): Model for embeddings
- `ie_model`, `et_model`, `ea_model`, `lp_model` (str, optional): Specific models for each pipeline component
- `similarity_threshold` (float, default 0.6): Threshold for entity similarity matching
- `output` (str, optional): File path to save JSON results

### Return Value

Returns a dictionary containing the complete CTI analysis results:

- `text`: The original input text
- `IE`: Intelligence Extraction results with:
  - `triplets`: Raw extracted subject-relation-object triplets
- `ET`: Entity Typing results with:
  - `typed_triplets`: Triplets with entity type classifications (Malware, Vulnerability, Infrastructure, etc.)
- `EA`: Entity Alignment results with:
  - `aligned_triplets`: Triplets with merged entities and canonical entity IDs
- `LP`: Link Prediction results with:
  - `predicted_links`: Additional predicted relationships between entities
- `entity_relation_graph`: File path to the interactive HTML visualization

---

<a id="local-setup"></a>

## 🐍 Local Development Setup

For users who want to run the web interface, use the command line interface, or contribute to the project, you'll need to clone the repository and set up the development environment.

### Prerequisites

- **API Key** from one of the supported providers: OpenAI, Gemini, AWS, or **Ollama** (local, free)
- **Python 3.11+** and pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/peng-gao-lab/CTINexus.git
cd CTINexus
```

### Step 2: Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your API credentials.

> **Note**: You only need to set up one provider. If using Ollama, see the [Ollama Guide](docs/ollama-guide.md).

### Step 3: Setup Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (macOS/Linux:)
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install package
pip install -e .
```

### Step 4: Run the Application

```bash
ctinexus
```

### Step 5: Access the Application

Open your browser and navigate to: **http://127.0.0.1:7860**

Use `Ctrl+C` in the terminal to stop the application.

---

<a id="docker-setup"></a>

## 🐳 Docker Setup

For containerized deployment or development:

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- API keys configured (see Local Development Setup above)

### Step 1: Clone the Repository

```bash
git clone https://github.com/peng-gao-lab/CTINexus.git
cd CTINexus
```

### Step 2: Configure API Keys

Create a `.env` file as described in the Local Development Setup section.

### Step 3: Launch with Docker

```bash
# Build and start
docker compose up --build

# Or run in detached mode (runs in background)
docker compose up -d --build
```

### Step 4: Access the Application

Open your browser and navigate to: **http://localhost:8000**

### Step 5: Stop the Application

```bash
docker compose down
```

---

## Web Interface and CLI Usage

After setting up the local environment, you can use CTINexus through the web interface or command line.

### ⚡ Command Line Interface (CLI)

For automation and batch processing:

```bash
ctinexus --input-file report.txt
```

**📖 [Complete CLI Documentation](docs/cli-guide.md)** - Detailed usage examples and options.

### 🖥️ Web Interface (GUI)

Once the application is running:

1. **Open your browser** to the appropriate URL:

   - Docker: `http://localhost:8000`
   - Local: `http://127.0.0.1:7860`

1. **Paste threat intelligence text** into the input area

1. **Select your preferred AI model** from the dropdown

1. **Click "Run"** to analyze the text

1. **View results**:

   - **Extracted Entities**: Identified cybersecurity entities
   - **Relationships**: Discovered connections between entities
   - **Interactive Graph**: Network visualization
   - **Export Options**: Download results as JSON or images

## Contributing

We warmly welcome contributions from the community! Whether you're interested in:

- 🐛 **Fixing bugs** or adding new features
- 📖 **Improving documentation** or adding examples
- 🎨 **UI/UX enhancements** for the web interface

Please check out our **[Contributing Guide](CONTRIBUTING.md)** for detailed information on how to get started, development setup, and submission guidelines.

## Citation

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
