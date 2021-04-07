# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StopWordsRemover, HashingTF, IDF, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


spark = SparkSession.builder.appName("zybova_spark").getOrCreate()

my_schema = StructType([
    StructField(name='id', dataType=IntegerType(), nullable=True),
    StructField(name='fraudulent', dataType=StringType(), nullable=True),
    StructField(name='full_description', dataType=StringType(), nullable=True)])

# read the dataset
my_data = spark \
    .read \
    .format("csv") \
    .schema(my_schema) \
    .options(path="final_project", header=True) \
    .load()

label_stringIdx = StringIndexer(inputCol = 'fraudulent', outputCol = 'label')
my_data = label_stringIdx.fit(my_data).transform(my_data)
my_data.show()

stages = []
tokenizer = RegexTokenizer(inputCol= 'full_description' , outputCol= 'words', pattern= '\\W')
stopwords = StopWordsRemover(inputCol= 'words', outputCol= 'filtered_words')
hashingTF = HashingTF(inputCol='filtered_words', outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
stages += [tokenizer, stopwords,hashingTF,idf]

lr = LogisticRegression(featuresCol= 'features', labelCol= 'label')
stages += [lr]

pipeline = Pipeline().setStages(stages)

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(metricName= "areaUnderPR"),
                          numFolds=5)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(my_data)
cvModel.bestModel.write().overwrite().save("my_final_model")

