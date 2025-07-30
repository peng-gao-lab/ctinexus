<div align="center">
  <img src="app/static/logo.png" alt="Logo" width="200">
  <h1 align="center">Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models</h1>
</div>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lavender.svg" alt="License: MIT"></a>
  <a href='https://github.com/peng-gao-lab/CTINexus'><img src='https://img.shields.io/badge/Project-Github-pink'></a>
  <a href='https://arxiv.org/abs/2410.21060'><img src='https://img.shields.io/badge/Paper-Arxiv-crimson'></a>  
  <a href='https://ctinexus.github.io/' target='_blank'><img src='https://img.shields.io/badge/Project-Website-turquoise'></a>
</p>

**CTINexus** is a framework that leverages optimized in-context learning (ICL) of large language models (LLMs) for automatic cyber threat intelligence (CTI) knowledge extraction and cybersecurity knowledge graph (CSKG) construction. 
CTINexus adapts to various cybersecurity ontologies with minimal annotated examples and provides a user-friendly web interface for instant threat intelligence analysis. 

<p align="center">
  <img src="app/static/overview.png" alt="CTINexus Framework Overview" width="500"/>
</p>

### What CTINexus Does

The framework automatically processes unstructured threat intelligence reports to:
- **Extract cybersecurity entities** (malware, vulnerabilities, tactics, IOCs)
- **Identify relationships** between security concepts  
- **Construct knowledge graphs** with interactive visualizations
- **Require minimal configuration** - no extensive data or parameter tuning needed

### Core Components

* **Intelligence Extraction (IE)**: Automatically extracts cybersecurity entities and relationships from unstructured text using optimized prompt construction and demonstration retrieval
* **Hierarchical Entity Alignment**: Canonicalizes extracted knowledge and removes redundancy through:
   * **Entity Typing (ET)**: Groups mentions of the same semantic type
   * **Entity Merging (EM)**: Merges mentions referring to the same entity with IOC (Indicator of Compromise) protection
* **Link Prediction (LP)**: Predicts and adds missing relationships to complete the knowledge graph
* **Graph Visualization**: Interactive network visualization of the constructed cybersecurity knowledge graph

<p align="center">
  <img src="app/static/webui.png" alt="CTINexus WebUI" width="500"/>
</p>

## News

🌟 [2024/07/29] CTINexus now features an intuitive Gradio interface! Submit threat intelligence text and instantly visualize extracted interactive graphs.

🔥 [2025/04/21] We released the camera-ready paper on [arxiv](https://arxiv.org/pdf/2410.21060). 

🔥 [2025/02/12] CTINexus is accepted at 2025 IEEE European Symposium on Security and Privacy ([Euro S&P](https://eurosp2025.ieee-security.org/index.html)).


## Quick Start

You can use CTINexus in two ways:
- **⚡ Command Line**: For automation and batch processing → **[📖 CLI Guide](app/docs/cli-guide.md)**
- **🖥️ Web Interface**: User-friendly GUI for interactive analysis (follow the setup below)

Both options support Docker and local installation.

### Prerequisites

- **API Key** from one of the supported providers: OpenAI, Gemini, AWS

- **For Local Setup Only**: Python 3.11+ and pip
- **For Docker Setup Only**: [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed and running

<br>

**Step 1: Clone and Configure**

```bash
# Clone the repository
git clone https://github.com/peng-gao-lab/CTINexus.git
cd CTINexus

# Copy environment template
cp .env.example .env
```

**Step 2: Configure API Keys**

Edit the `.env` file with your API credentials:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

> **Note**: You only need to set up one provider, but you may configure multiple APIs if desired. Once configured, you can select models from any of your chosen providers.


---

<a id="local-setup"></a>

## 🐍 Using Local Setup

### Step 3: Setup Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app/app.py

# Or you can run from the app/ directory:
# cd app
# python app.py
```

### Step 5: Access the Application

Open your browser and navigate to: **http://127.0.0.1:7860**

### Step 6: Stop the Application

Press `Ctrl+C` in the terminal to stop the application.

---

<a id="docker-setup"></a>

## 🐳 Using Docker


### Step 3: Launch with Docker

```bash
# Build and start
docker-compose up --build

# Or run in detached mode (runs in background)
docker-compose up -d --build
```

### Step 4: Access the Application

Open your browser and navigate to: **http://localhost:8000**

### Step 5: Stop the Application

```bash
# Stop the application
docker-compose down
```

---

## Using CTINexus

### ⚡ Command Line Interface (CLI)

For automation, batch processing, or integration into existing workflows, use the CLI:

```bash
python app.py --input-file report.txt
```

**📖 [Complete CLI Documentation](app/docs/cli-guide.md)** - Detailed usage examples, model options, and integration guides.

### 🖥️ Web Interface (GUI)

Once the application is running (either via Docker or locally):

1. **Open your browser** to the appropriate URL:
   - Docker: `http://localhost:8000`
   - Local: `http://127.0.0.1:7860`

2. **Paste threat intelligence text** into the input area (e.g., security reports, vulnerability descriptions, incident reports)

3. **Select your preferred AI model** from the dropdown

4. **Click "Run"** to analyze the text

5. **View results**:
   - **Extracted Entities**: Identified cybersecurity entities (malware, vulnerabilities, tactics, etc.)
   - **Relationships**: Discovered connections between entities
   - **Interactive Graph**: Network visualization of the knowledge graph
   - **Export Options**: Download results as JSON or graph images


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