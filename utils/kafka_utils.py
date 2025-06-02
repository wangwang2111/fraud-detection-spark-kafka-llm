# Kafka streamline lib
# pip install confluent-kafka python-dotenv
from confluent_kafka import Consumer, Producer
from dotenv import load_dotenv
import os
import json

# Load environment variables (Kafka config)
load_dotenv()

def get_kafka_consumer():
    """Initialize and return a Kafka consumer for real-time message processing."""
    kafka_config = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': os.getenv('KAFKA_CONSUMER_GROUP', 'dialogue-classifier-group'),
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False  # Manual commit after processing
    }

    # Optional SASL/SSL authentication (if required)
    if os.getenv('KAFKA_SECURITY_PROTOCOL') == 'SASL_SSL':
        kafka_config.update({
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': os.getenv('KAFKA_USERNAME'),
            'sasl.password': os.getenv('KAFKA_PASSWORD')
        })

    consumer = Consumer(kafka_config)
    consumer.subscribe([os.getenv('KAFKA_INPUT_TOPIC', 'customer-dialogues-raw')])
    return consumer

def get_kafka_producer():
    """Initialize and return a Kafka producer for sending processed results."""
    kafka_config = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    }

    # Optional SASL/SSL authentication (if required)
    if os.getenv('KAFKA_SECURITY_PROTOCOL') == 'SASL_SSL':
        kafka_config.update({
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': os.getenv('KAFKA_USERNAME'),
            'sasl.password': os.getenv('KAFKA_PASSWORD')
        })

    producer = Producer(kafka_config)
    return producer