from kafka import KafkaProducer
import json
import csv
import sys

TOPIC_NAME = 'email'
KAFKA_SERVER = 'localhost:9092'
csv_file = 'testData.csv'

producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER)

maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Read CSV file and publish each row to Kafka topic
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Convert CSV row to JSON
        message = json.dumps(row).encode('utf-8')
        producer.send(TOPIC_NAME, value=message)
        print(f"Published: {message}")
        producer.flush()