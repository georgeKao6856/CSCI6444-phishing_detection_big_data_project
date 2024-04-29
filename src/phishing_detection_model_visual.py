from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, NGram
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder \
    .appName("phishing_detction_model") \
    .getOrCreate()


schema = StructType([
    StructField("_c0", StringType(), True),
    StructField("Email Text", StringType(), True),
    StructField("Email Type", StringType(), True)
])


df = spark.read.csv("Phishing_Email.csv", schema=schema, header=True, multiLine=True, quote='"', sep=",", escape='"')
df.show()

df = df.na.drop(subset=["Email Text"])
df = df.withColumnRenamed("_c0", "emailID")
df.show()

##Pie chart for phishing and nonphishing emails
total_emails = df.count()
phishing_emails = df.filter(df['Email Type'] == 'Phishing Email').count()
non_phishing_emails = total_emails - phishing_emails

phishing_percentage = (phishing_emails / total_emails) * 100
non_phishing_percentage = (non_phishing_emails / total_emails) * 100

# Data to plot
labels = 'Phishing', 'Non-Phishing'
sizes = [phishing_percentage, non_phishing_percentage]
colors = ['gold', 'lightskyblue']

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Adding a title
plt.title('Percentage of Phishing and Non-Phishing Emails')
plt.savefig('pictures/pie-chart-phishing.png')

# Display the chart
#plt.show()


#df = df.limit(1000)

##Word Cloud visualization before model training

phishing_texts = df.filter(df['Email Type'] == 'Phishing Email').select('Email Text')
phishing_texts_list = [row['Email Text'] for row in phishing_texts.collect()]
phishing_text = ' '.join(phishing_texts_list)
# Generate the word cloud
phishing_cloud = WordCloud(width=800, height=500, background_color='white', max_words=50).generate(phishing_text)

# Display the word cloud
plt.figure(figsize=(10, 8), facecolor='b')
plt.imshow(phishing_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('pictures/word_cloud_dataset.png')
#plt.show()

trainingData, testData = df.randomSplit([0.7, 0.3])
testData.show()

#testData.write.csv("testData", header = True)
testData.select('emailID','Email Text','Email Type').toPandas().to_csv('testData.csv',index=False)

##Text preprocessing
tokenizer = Tokenizer().setInputCol('Email Text').setOutputCol('words')
stopwords = StopWordsRemover().getStopWords() + ['-']
remover = StopWordsRemover().setStopWords(stopwords).setInputCol('words').setOutputCol('filtered')
bigram = NGram().setN(2).setInputCol('filtered').setOutputCol('bigrams')
cvmodel = CountVectorizer().setInputCol('filtered').setOutputCol('features')
cvmodel_ngram = CountVectorizer().setInputCol('bigrams').setOutputCol('features')
indexer = StringIndexer().setInputCol('Email Type').setOutputCol('label') 

nb = NaiveBayes(smoothing=1)
pipeline = Pipeline(stages = [tokenizer, remover, cvmodel, indexer, nb])

model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select('Email Text', 'label', 'rawPrediction', 'probability', 'prediction').show(5)

##Confusion matrix visualization
actual_labels = predictions.select('label').rdd.flatMap(lambda x: x).collect()
predicted_labels = predictions.select('prediction').rdd.flatMap(lambda x: x).collect()

# Create a confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Phishing', 'Phishing'], yticklabels=['Not Phishing', 'Phishing'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.savefig('pictures/confusion_matrix.png')
#plt.show()

##Word Cloud for Model Predictions
# Filter the predictions DataFrame to select only phishing emails
phishing_texts = predictions.filter(predictions['prediction'] == 1).select('Email Text')

# Collect the phishing emails into a list
phishing_texts_list = [row['Email Text'] for row in phishing_texts.collect()]

# Concatenate the phishing emails into a single string
phishing_text = ' '.join(phishing_texts_list)

# Generate the word cloud
phishing_cloud = WordCloud(width=800, height=500, background_color='white', max_words=50).generate(phishing_text)

# Display the word cloud
plt.figure(figsize=(10, 8), facecolor='b')
plt.imshow(phishing_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('pictures/word_cloud_model.png')
##plt.show()


"""
##AUC Score
evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName('areaUnderROC')
AUC = evaluator.evaluate(predictions)
print("AUC:", AUC*100)
"""

predictionAndLabels = predictions.select("prediction", "label").rdd
metrics = MulticlassMetrics(predictionAndLabels)
# Calculate accuracy, precision, recall, and F1 score
accuracy = metrics.accuracy
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1Score = metrics.fMeasure(1.0)
# Print the metrics
print(f"Accuracy: {accuracy*100}")
print(f"Precision: {precision*100}")
print(f"Recall: {recall*100}")
print(f"F1 Score: {f1Score*100}")

model_path = "naive_bayes_model"
model.save(model_path)

# Stop Spark session
spark.stop()

