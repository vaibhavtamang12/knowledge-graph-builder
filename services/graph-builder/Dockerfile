FROM python:3.11-slim

WORKDIR /app

COPY services/graph-builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/graph-builder/ .

CMD ["python", "main.py"]