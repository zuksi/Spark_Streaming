# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("zybova_spark").getOrCreate()

new_schema = StructType([
    StructField(name='id', dataType=IntegerType(), nullable=True),
    StructField(name='company_id', dataType=IntegerType(), nullable=True),
    StructField(name='full_description', dataType=StringType(), nullable=True)])

new_ads = spark \
    .readStream \
    .format("csv") \
    .schema(new_schema) \
    .options(path="fp_stream", header=True,maxFilesPerTrigger=1) \
    .load()


def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq) \
        .options(truncate=True) \
        .start()

s = console_output(new_ads, 5)
s.stop()

cassandra_companies = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="fraudulent_companies", keyspace="zybova") \
    .load()

cassandra_companies.show()

cvModel = PipelineModel.load("my_final_model")

# вся логика в этом foreachBatch
def writer_logic(df, epoch_id):
    df.persist()
    print("---------I've got new batch--------")
    print("New_ads:")
    df.show()
    cassandra_companies.persist()
    print("Here is what I've got from Cassandra:")
    cassandra_companies.show()
    cassandra_companies_df = cassandra_companies.select("company_id").distinct()
    cassandra_companies_list_rows = cassandra_companies_df.collect()
    cassandra_companies_list = map(lambda x: x.__getattr__("company_id"), cassandra_companies_list_rows)
    list_companies = map(lambda row: row.asDict(), cassandra_companies.collect())
    dict_companies = {company['company_id']: company['fake_ads'] for company in list_companies}
    companies_list_rows_df = df.select("company_id").distinct().collect()
    companies_list_df = map(lambda x: x.__getattr__("company_id"), companies_list_rows_df)
    for company in companies_list_df:
        if company in cassandra_companies_list and dict_companies[company] > 1:
            cassandra_new = cassandra_companies.select(cassandra_companies['company_id'],
                                                       cassandra_companies['fake_ads'] + 1) \
                .where(F.col('company_id') == company)
            cassandra_new = cassandra_new.select(F.col('company_id'), F.col('(fake_ads + 1)').alias('fake_ads'))
            cassandra_new.show()
            cassandra_new.write \
                .format("org.apache.spark.sql.cassandra") \
                .options(table="fraudulent_companies", keyspace="zybova") \
                .mode("append") \
                .save()
            print("I updated data in Cassandra. Continue...")
            df = df.where(F.col('company_id') != company)
    predict = cvModel.transform(df).select('id', 'company_id', 'full_description', 'prediction')
    print("Predictions:")
    predict.show()
    predict_fake = predict.where((F.col('prediction') == 1.0) & (F.col('company_id') != 1710))
    predict_fake.show()
    predict_short = predict_fake.select(F.col("company_id"), F.col("prediction").alias("fake_ads"))
    predict_short.show()
    cassandra_stream_union = predict_short.union(cassandra_companies)
    cassandra_stream_aggregation = cassandra_stream_union.groupBy("company_id").agg(F.sum("fake_ads").alias("fake_ads"))
    print("Aggregated data:")
    cassandra_stream_aggregation.show()
    cassandra_stream_aggregation.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="fraudulent_companies", keyspace="zybova") \
        .mode("append") \
        .save()
    print("I saved the aggregation in Cassandra. Continue...")
    cassandra_companies.unpersist()
    df.unpersist()


stream = new_ads \
    .writeStream \
    .trigger(processingTime='100 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/fake_ads_checkpoint")

# поехали
s = stream.start()

s.stop()