# ====================================================================
# spark_pipeline_fixed.py
# ====================================================================

import os
import time
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, to_timestamp, avg, sum as sum_, count, when
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ====================================================================
# PATHS
# ====================================================================
BRONZE_PATH = "/app/storage/bronze_stock"
SILVER_PATH = "/app/storage/silver_stock"
GOLD_PATH = "/app/storage/gold_stock"

CKPT_BRONZE = "/app/storage/ckpt/bronze"
CKPT_SILVER = "/app/storage/ckpt/silver"
CKPT_GOLD = "/app/storage/ckpt/gold"

for p in [BRONZE_PATH, SILVER_PATH, GOLD_PATH, CKPT_BRONZE, CKPT_SILVER, CKPT_GOLD]:
    os.makedirs(p, exist_ok=True)


# ====================================================================
# SPARK
# ====================================================================
spark = (
    SparkSession.builder
        .appName("StockPipeline")
        .master("local[*]")
        .config("spark.jars.packages",
                "io.delta:delta-core_2.12:2.4.0,"
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
                "org.apache.kafka:kafka-clients:3.5.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
)


# ====================================================================
# SCHEMA
# ====================================================================
schema = StructType([
    StructField("timestamp", StringType()),
    StructField("symbol", StringType()),
    StructField("price", DoubleType()),
    StructField("volume", IntegerType()),
    StructField("rsi", DoubleType()),
    StructField("macd", DoubleType()),
    StructField("volatility", DoubleType()),
    StructField("price_change", DoubleType()),
    StructField("price_direction", IntegerType()),
    StructField("volume_change", IntegerType())
])

batch_counter = 0


# ====================================================================
# SILVER BATCH PROCESSOR (SAVE SILVER + ML + GOLD)
# ====================================================================
def process_silver_batch(df, batch_id):
    global batch_counter
    batch_counter += 1

    print(f"\n=== SILVER BATCH {batch_counter} ===")

    df.persist()
    n = df.count()
    print("Batch size:", n)

    # ----------------------------------------------------------
    # 1) SAVE SILVER DATA (MAIN FIX)
    # ----------------------------------------------------------
    try:
        df.write.format("delta") \
            .mode("append") \
            .save(SILVER_PATH)

        print("Silver saved ✓")

    except Exception as e:
        print("Silver Save Error:", e)

    # ----------------------------------------------------------
    # 2) TRAIN ML EVERY 5 BATCHES
    # ----------------------------------------------------------
    if batch_counter % 5 == 0 and n > 300:
        try:
            print("Training ML model...")

            features = ["rsi", "macd", "volatility", "price_change", "volume_change"]
            assembler = VectorAssembler(inputCols=features, outputCol="features")

            ml_df = assembler.transform(df).select("features", "price_direction")

            train, test = ml_df.randomSplit([0.8, 0.2], 42)

            model = RandomForestClassifier(
                featuresCol="features",
                labelCol="price_direction",
                numTrees=40,
                maxDepth=6,
                seed=42
            ).fit(train)

            preds = model.transform(test)

            acc = MulticlassClassificationEvaluator(
                labelCol="price_direction",
                predictionCol="prediction",
                metricName="accuracy"
            ).evaluate(preds)

            print(f"Accuracy = {acc:.4f}")

        except Exception as e:
            print("ML Error:", e)

    # ----------------------------------------------------------
    # 3) GOLD KPIs
    # ----------------------------------------------------------
    try:
        print("Updating GOLD KPIs...")

        df_gold = df.groupBy("symbol").agg(
            avg("price").alias("avg_price"),
            sum_("volume").alias("total_volume"),
            avg("volatility").alias("avg_volatility"),
            (
                sum_(when(col("price_direction") == 1, 1).otherwise(0)) /
                count("*") * 100
            ).alias("pct_upward")
        )

        df_gold.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(GOLD_PATH)

        print("Gold updated ✓")

    except Exception as e:
        print("Gold Error:", e)

    df.unpersist()


# ====================================================================
# MAIN STREAMING PIPELINE
# ====================================================================
def start_pipeline():

    try:
        print("Starting streams...")

        # -------------------- BRONZE -------------------------
        kafka_df = (
            spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "kafka:9092")
                .option("subscribe", "stock_fin")
                .option("startingOffsets", "earliest")
                .option("failOnDataLoss", "false")
                .load()
        )

        bronze_df = kafka_df.selectExpr("CAST(value AS STRING) AS json") \
            .select(from_json(col("json"), schema).alias("data")).select("data.*")

        bronze_df.writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", CKPT_BRONZE) \
            .start(BRONZE_PATH)

        # -------------------- SILVER -------------------------
        silver_df = (
            bronze_df
                .withColumn("timestamp", to_timestamp(col("timestamp")))
                .filter(col("symbol").isNotNull())
                .filter(col("price") > 0)
                .filter(col("volume") >= 0)
        )

        silver_df.writeStream \
            .foreachBatch(process_silver_batch) \
            .option("checkpointLocation", CKPT_SILVER) \
            .trigger(processingTime="5 seconds") \
            .start()

        print("Pipeline Running...")
        spark.streams.awaitAnyTermination()

    except Exception as e:
        print("Stream Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    time.sleep(5)
    start_pipeline()
