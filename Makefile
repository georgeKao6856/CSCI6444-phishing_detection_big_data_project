FILE_URL := https://github.com/docker/compose/releases/latest/download/docker-compose-Linux-x86_64

All: install_docker install_docker_compose install_py_packages create_folder hdfs kafka_setup

kafka_setup:
	sudo dnf install git-all -y
	git clone https://github.com/conduktor/kafka-stack-docker-compose
	sed -i '9s/.*/      - "2185:2181"/' kafka-stack-docker-compose/zk-single-kafka-single.yml
	sudo docker-compose -f kafka-stack-docker-compose/zk-single-kafka-single.yml up

install_py_packages:
	pip install kafka-python
	pip install pandas
	pip install numpy
	pip install pyspark
	pip install wordcloud
	pip install scikit-learn
	pip install seaborn

hdfs:
	hdfs dfs -put Phishing_Email.csv

install_docker:
	sudo yum install docker -y
	sudo docker --version

install_docker_compose:
	sudo curl -L $(FILE_URL) -o /usr/local/bin/docker-compose
	sudo chmod +x /usr/local/bin/docker-compose
	docker-compose version

create_folder:
	mkdir pictures