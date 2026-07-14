# Databricks notebook source
# MAGIC %md
# MAGIC # DATA 612 Final Project — Secondary Metrics + Qualifying Predictions
# MAGIC
# MAGIC Runs against the SAVED winning artifacts from the sweep (no refitting --
# MAGIC the same train-once/evaluate-many pattern as Project 6's deployment):
# MAGIC
# MAGIC 1. precision@k / recall@k / catalog coverage on the probe predictions --
# MAGIC    the planning document's check on whether RMSE gains translate into
# MAGIC    recommendations that actually look better.
# MAGIC 2. A qualifying.txt-formatted prediction file, the original competition's
# MAGIC    submission format (unscorable -- Netflix never released the answer key --
# MAGIC    but it demonstrates the pipeline end to end in the Prize's own terms).

# COMMAND ----------

import os
import sys
import time

# insert(0), not append: the ML runtime pre-installs HuggingFace's `evaluate`
# package, which shadows this pipeline's evaluate.py unless our folder wins the
# sys.path race.
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))

from pyspark.sql import functions as F

from parser import to_spark_df, load_probe_pairs
from temporal_bias import apply_bias, clip_to_scale
from train_als import load_pipeline_artifacts
from evaluate import split_train_probe, score_predictions
from metrics import precision_recall_at_k, catalog_coverage
from qualifying_writer import generate_qualifying_predictions

# COMMAND ----------

PARQUET_PATH = "dbfs:/netflix_prize_data/parquet"
DATA_PATH = "/dbfs/netflix_prize_data"
ARTIFACTS_PATH = "/dbfs/data612_final_project/sweep_artifacts_time"  # time-split winners
QUALIFYING_OUT = "/dbfs/data612_final_project/qualifying_predictions.txt"
CATALOG_SIZE = 17_770  # full Netflix Prize catalog

t_start = time.time()

# COMMAND ----------

# MAGIC %md ## Load saved artifacts + rebuild probe predictions

# COMMAND ----------

als_model, bias_model = load_pipeline_artifacts(spark, ARTIFACTS_PATH)
ratings_df = spark.read.parquet(PARQUET_PATH)
probe_pairs_df = to_spark_df(spark, load_probe_pairs(DATA_PATH))
_, probe_truth_df = split_train_probe(ratings_df, probe_pairs_df)
probe_truth_df = probe_truth_df.cache()

t0 = time.time()
probe_bias = apply_bias(probe_truth_df, bias_model)
preds = als_model.transform(probe_bias)
preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
preds = clip_to_scale(preds, "raw_pred", "pred").cache()
scores = score_predictions(preds, "rating", "pred")
print(f"probe RMSE={scores['rmse']:.4f} MAE={scores['mae']:.4f} "
      f"({scores['n_scored']:,} rows, {time.time() - t0:.1f}s)")

# COMMAND ----------

# MAGIC %md ## Precision@k / recall@k / coverage

# COMMAND ----------

t0 = time.time()
for k in (5, 10):
    pr = precision_recall_at_k(preds, k=k, threshold=4.0)
    print(f"k={k}: precision@k={pr['precision_at_k']:.4f} recall@k={pr['recall_at_k']:.4f} "
          f"(users scored: {pr['n_users_scored']:,})")
cov = catalog_coverage(preds, catalog_size=CATALOG_SIZE, k=10)
print(f"coverage@10: {cov['movies_recommended']:,}/{cov['catalog_size']:,} = {cov['coverage']:.2%}")
print(f"metrics took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Qualifying prediction file

# COMMAND ----------

t0 = time.time()
n = generate_qualifying_predictions(spark, DATA_PATH, als_model, bias_model, QUALIFYING_OUT)
print(f"wrote {n:,} predictions to {QUALIFYING_OUT} in {time.time() - t0:.1f}s")
print(f"\nTOTAL: {time.time() - t_start:.1f}s")
