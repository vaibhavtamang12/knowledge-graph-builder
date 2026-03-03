from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import json
import time

def get_consumer(retries=10, delay=5):
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                "nlp-processed",
                bootstrap_servers="kafka:9092",
                group_id="graph-builder-group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True
            )
            print("Connected to Kafka successfully.")
            return consumer
        except NoBrokersAvailable:
            print(f"Kafka not ready, retrying in {delay}s... (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
    raise Exception("Could not connect to Kafka after multiple retries.")

def get_messages():
    consumer = get_consumer()
    for message in consumer:
        yield message.value