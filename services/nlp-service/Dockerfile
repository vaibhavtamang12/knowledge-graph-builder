FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/nlp-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy English model used by extractor.py
RUN python -m spacy download en_core_web_sm

COPY services/nlp-service/ .

CMD ["python", "main.py"]