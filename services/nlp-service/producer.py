from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import time

_producer = None

def get_producer(retries=10, delay=5):
    global _producer
    if _producer is not None:
        return _producer
    for attempt in range(retries):
        try:
            _producer = KafkaProducer(
                bootstrap_servers="kafka:9092",
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            print("Kafka producer connected successfully.")
            return _producer
        except NoBrokersAvailable:
            print(f"Kafka not ready, retrying in {delay}s... (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    raise Exception("Could not connect to Kafka producer after multiple retries.")

def send_processed(message: dict):
    producer = get_producer()
    producer.send("nlp-processed", message)
    producer.flush()