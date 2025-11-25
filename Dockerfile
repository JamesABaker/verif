# Use Python slim image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/

# Build arg to control PyTorch variant (cpu or cu118)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install dependencies with uv
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install torch --index-url ${TORCH_INDEX_URL} && \
    uv pip install -e .

# Set environment variables to use venv
ENV PATH=/app/.venv/bin:$PATH
ENV VIRTUAL_ENV=/app/.venv

# Pre-download the model during build to avoid startup delays
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); \
    AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
