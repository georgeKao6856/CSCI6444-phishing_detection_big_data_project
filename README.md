# CSCI 6444 - Phishing Email Detection Project

## Introduction
Welcome to the Phishing Email Detection User Manual. Our system provides real-time detection capabilities to identify phishing emails and promptly alerts managers. Additionally, it is equipped to efficiently handle large volumes of data. Please follow this manual's guidance to install our system step-by-step.

## System Requirements
Hardware Requirements:
* Minimum 4 Core CPU
* Minimum16 GB memory
* 10 GB of available disk space

Software Requirements:
* Operating System: Windows 10, macOS, or Linux
* Python 3.x
* Hadoop 3.3.6
* Spark 3.5.0

Recommend Using:
* AWS EMR

## System Architecture
We used AWS EMR to create a cluster environment that contains Hadoop and Spark. Then, we installed the kafka and zookeeper in the docker containers. 

![System Architecture](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/System%20Architecture.png)

## Data Flow
* All the data will be stored in the Hadoop HDFS.

![Data Flow](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/DataFlow.png)

## Installation
1. Download the Phishing Email Detection software from our GitHub. https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project 
2. In AWS EMR, Create a cluster.
3. According to the machine’s IP address, upload Makefile, phishing_consumer.py, phishing_detection_model_visual.py, phishing_listener.py, phishing_producer.py, and Phishing_Email.csv to the master node.
4. Connect to the master node and open a terminal.
5. Run the command ‘make’ in the terminal, it will install every necessary dependencies and start the Kafka service.

## Getting Started
1.	Training the model:
    `spark-submit phishing_detection_model_visual.py`
2.	Start the phishing email listener:
    `phishing_listener.py`
3.	Start the Kafka consumer:
    `spark-submit phishing_consumer.py`
4.	Start the Kafka producer:
    `python phishing_producer.py`

## Output
* testData.csv – The data in this file is for simulating the new email that comes into our system.
* phishing_email.txt – This file contains all the phishing emails our system detected.
* confusion_matrix.png – This diagram displays true positives, true negatives, false positives and false negatives.

![confusion_matrix](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/confusion_matrix.png)

* pie-chart-phishing.png – This diagram displays the percentage of phishing and non-phishing email in the dataset.

![pie-chart-phishing](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/pie-chart-phishing.png)

* word_cloud_dataset.png – This diagram displays the most frequent words found in all phishing emails in the dataset.

![word_cloud_dataset](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/word_cloud_dataset.png)

* word_cloud_model.png – This diagram displays the most frequent words found in all phishing emails that found by our system.

![word_cloud_model](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/word_cloud_model.png)

* cpu_utilization.jpg – This diagram displays the CPU utilization when you train the model.

![cpu_utilization](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/cpu_utilization.jpg)

## Result

* Naive Bayes Model

![model_statistic](https://github.com/georgeKao6856/CSCI6444-phishing_detection_big_data_project/blob/main/pictures/model_statistic.png)
