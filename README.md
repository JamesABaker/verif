# verif

[![Docker Build](https://github.com/JamesABaker/verif/actions/workflows/docker.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/docker.yml)
[![pre-commit](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml)

A hybrid AI text detection system combining machine learning with information theory. Uses RoBERTa ML classifier + entropy-based analysis (perplexity, burstiness, Shannon entropy) for robust detection of AI-generated text.

## Features

- 🧠 **Hybrid Detection** - combines ML model with entropy analysis for broader insights
- 📊 **Information Theory** - perplexity, Shannon entropy, burstiness, lexical diversity
- 🎯 **Robust Against Modern LLMs** - entropy features work on GPT-4/GPT-5 output
- 🐳 **Fully containerized** - runs anywhere with Docker
- 🌐 **Web UI + REST API** - easy to use, easy to integrate
- 🚀 **Fast inference** - results in seconds
- 💾 **Model caching** - downloads once, runs forever
- 🔒 **Privacy-first** - runs completely locally, no external API calls

## Quick Start

### Run with Docker Compose (Recommended)

Install [Docker](https://docs.docker.com/get-docker/) if you haven't already.


### Run with Docker (Manual)

```bash
# Build the image
docker build -t verif .

# Run the container
docker run -p 8000:8000 verif
```

## Usage

### Web Interface

1. Open http://localhost:8000 in your browser
2. Paste or type text into the textarea
3. Click "Detect AI"
4. View the results

### REST API

**Detect Text:**
```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze here..."}'
```

**Response:**
```json
{
  "human_probability": 45.3,
  "ai_probability": 54.7,
  "prediction": "ai",
  "text_length": 123
}
```

**Get Model Info:**
```bash
curl http://localhost:8000/api/model-info
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### Interactive API Documentation

Visit http://localhost:8000/docs for interactive API documentation with a built-in testing interface.

## Evaluation

To evaluate the detection system against the HC3 dataset (human vs ChatGPT text), first ensure the API is running, then build and run the evaluation Docker container:

```bash
# Download the dataset once (this caches it locally in /data/hc3_dataset)
python scripts/download_hc3.py

# Then run evaluation
docker build -f Dockerfile.eval -t verif-eval .
docker run --network host -v hc3-cache:/data/hc3_dataset -e HC3_CACHE_DIR=/data/hc3_dataset verif-eval
```

This will test 1000 human and 1000 AI samples from the HC3 dataset and output accuracy metrics to `results.json`.

## Development

### Local Development Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

**Quick Start:**

```bash
# Create a virtual environment and install all dependencies
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

**Selective Installation:**

```bash
# Just app dependencies (no dev or eval)
uv sync

# App + dev dependencies
uv sync --extra dev

# App + eval dependencies
uv sync --extra eval

# All dependencies
uv sync --all-extras
```

## Model Information

- **ML Component**: RoBERTa-base trained on ChatGPT output
- **Entropy Features**:
  - **Perplexity**: Measures text predictability (lower = more AI-like)
  - **Shannon Entropy**: Character-level randomness measure
  - **Burstiness**: Sentence complexity variation (human writing varies more)
  - **Lexical Diversity**: Unique word ratio (type-token ratio)


## Future Enhancements

- [ ] Batch processing API endpoint
- [ ] API rate limiting
- [ ] User 2fa authentication
- [ ] Method refinement
- [ ] Deployment
- [ ] Volume data persistence for users


## License

MIT License - feel free to use this project for learning, development, or production.
