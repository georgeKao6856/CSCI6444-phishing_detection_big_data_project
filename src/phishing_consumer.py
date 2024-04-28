from kafka import KafkaConsumer
import socket

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

SERVER_HOST = 'localhost'
SERVER_PORT = 5678

TOPIC_NAME = 'email'

consumer = KafkaConsumer(TOPIC_NAME)


spark = SparkSession.builder \
    .appName("detection_phishing_model") \
    .getOrCreate()

model_path = "naive_bayes_model"
model = PipelineModel.load(model_path)

for message in consumer:
    message_decoded = str(message.value, encoding='utf-8')
    #print(message_decoded)
    message_decoded = message_decoded.replace("\u0001", "")
    
    df = spark.read.json(spark.sparkContext.parallelize([message_decoded]))  # Fix the missing closing parenthesis and wrap message_decoded in a list
    #df.show()
    df = df.na.drop(subset=["Email Text"])

    predictions = model.transform(df)
    predictions.select('Email Text', 'label', 'rawPrediction', 'probability', 'prediction').show()

    pred = list(predictions.select('prediction').toPandas()['prediction'])[0]
    email_text = list(predictions.select('Email Text').toPandas()['Email Text'])[0]

    if(pred == 1.0):
        print("Phishing email found!")
        print('Email Text: ', predictions.select('Email Text'))

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        client_socket.sendall(email_text.encode('utf-8'))

        response = client_socket.recv(1024)
        print("Response from server:", response.decode())

        client_socket.close()




