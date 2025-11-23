# verif

[![Docker Build](https://github.com/JamesABaker/verif/actions/workflows/docker.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/docker.yml)
[![pre-commit](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/JamesABaker/verif/actions/workflows/pre-commit.yml)

A hybrid AI text detection system combining machine learning with information theory. Uses RoBERTa ML classifier + entropy-based analysis (perplexity, burstiness, Shannon entropy) for robust detection of AI-generated text.

## Features

- 🧠 **Hybrid Detection** - combines ML model with entropy analysis for better accuracy
- 📊 **Information Theory** - perplexity, Shannon entropy, burstiness, lexical diversity
- 🎯 **Robust Against Modern LLMs** - entropy features work on GPT-4/GPT-5 output
- 🐳 **Fully containerized** - runs anywhere with Docker
- 🌐 **Web UI + REST API** - easy to use, easy to integrate
- 🚀 **Fast inference** - results in seconds
- 💾 **Model caching** - downloads once, runs forever
- 🔒 **Privacy-first** - runs completely locally, no external API calls

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- 2GB free disk space (for model download)

### Run with Docker Compose (Recommended)

```bash
# Clone or navigate to the project directory
cd verif

# Start the application
docker-compose up

# Or run in detached mode
docker-compose up -d
```

The application will be available at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Run with Docker (Manual)

```bash
# Build the image
docker build -t verif .

# Run the container
docker run -p 8000:8000 -v model-cache:/root/.cache/huggingface verif
```

## Usage

### Web Interface

1. Open http://localhost:8000 in your browser
2. Paste or type text into the textarea
3. Click "Detect AI"
4. View the results showing human vs. AI probabilities

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

## Development

### Local Development Setup

**Using Conda**
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate verif

# Install pre-commit hooks
pre-commit install

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development setup instructions.


## Model Information

- **ML Model**: [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta)
- **Architecture**: Hybrid - RoBERTa + Entropy Analysis
- **ML Component**: RoBERTa-base trained on ChatGPT output
- **Entropy Features**:
  - **Perplexity**: Measures text predictability (lower = more AI-like)
  - **Shannon Entropy**: Character-level randomness measure
  - **Burstiness**: Sentence complexity variation (human writing varies more)
  - **Lexical Diversity**: Unique word ratio (type-token ratio)

## Technical Details

### Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **ML Framework**: PyTorch + Transformers
- **ML Model**: RoBERTa-base (ChatGPT detector)
- **Perplexity Model**: GPT-2 (for entropy calculation)
- **Entropy Analysis**: NumPy for statistical metrics
- **Container**: Docker with multi-stage build
- **Frontend**: Vanilla HTML/CSS/JavaScript

### Resource Requirements

- **RAM**: ~2GB minimum (for model loading)
- **Disk**: ~2GB (for model cache)
- **CPU**: Any modern CPU (GPU not required)

### Performance

- **First request**: 5-10 seconds (model loading)
- **Subsequent requests**: <1 second
- **Concurrent requests**: Supported (model is loaded once)

## Stopping the Application

```bash
# If running with docker-compose
docker-compose down

# To also remove the model cache volume
docker-compose down -v
```

## Future Enhancements

- [ ] Batch processing API endpoint
- [ ] Confidence threshold settings
- [ ] Text highlighting for AI-likely sections
- [ ] Export results to PDF/CSV
- [ ] API rate limiting
- [ ] User authentication
- [ ] Method refinement and white paper

## License

MIT License - feel free to use this project for learning, development, or production.

## Acknowledgments

- ML Model: [ChatGPT Detector RoBERTa](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) by Hello-SimpleAI
- Perplexity Analysis: GPT-2 by OpenAI
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Entropy-based detection inspired by information theory research

## Contributing

This is a learning/warmup project, but contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your use cases

---

**Note**: This detector identifies *style patterns* associated with AI writing. It should be used as one tool among many for academic integrity, not as definitive proof.
