FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by spaCy + transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/nlp-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models:
#   en_core_web_trf  – transformer-based model for high-quality NER + dep parse
#   en_core_web_sm   – lightweight fallback (extractor.py auto-selects)
RUN python -m spacy download en_core_web_trf
RUN python -m spacy download en_core_web_sm

COPY services/nlp-service/ .

CMD ["python", "main.py"]