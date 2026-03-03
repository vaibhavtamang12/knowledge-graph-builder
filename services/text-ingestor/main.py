from producer import send_message
from datetime import datetime

TOPIC = "raw-text-stream"

def run():
    sample_text = {
        "source": "sample-news",
        "text": "Elon Musk is the CEO of Tesla.",
        "timestamp": datetime.utcnow().isoformat()
    }

    send_message(TOPIC, sample_text)
    print("Message sent to Kafka!")

if __name__ == "__main__":
    run()