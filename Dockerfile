# VerifAI — container image
#
# Builds an image that can run any repo entry point; the default command is
# the Milestone 4 model pipeline (src/model_runner.py).
#
# Build:
#   docker build -t verifai .
#
# Run the model pipeline (writes outputs/samples.txt inside the container;
# mount outputs/ to keep the results):
#   docker run --env-file .env -v "$(pwd)/outputs:/app/outputs" verifai
#
# Run the API + UI instead:
#   docker run --env-file .env -p 8000:8000 verifai \
#     uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# Notes:
# - API keys are read from .env (see .env.template). Without an
#   ANTHROPIC_API_KEY the pipeline still runs, in classifier-fallback mode.
# - The image is large (~several GB) because of PyTorch.

FROM python:3.13-slim

WORKDIR /app

# Build tools needed by some scientific-Python wheels' fallbacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so Docker layer caching survives code edits
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (checkpoint included — the repo tracks best_model.pt)
COPY . .

CMD ["python", "src/model_runner.py"]
