# Databricks notebook source
# MAGIC %md
# MAGIC # DATA 612 Final Project — Ablation Ladder Pipeline
# MAGIC
# MAGIC Runs the full ablation ladder from the planning document against the real
# MAGIC Netflix Prize corpus, with wall-clock timing and both RMSE and MAE at every
# MAGIC rung, all scored on the actual probe.txt held-out split:
# MAGIC
# MAGIC 1. Global mean (sanity baseline)
# MAGIC 2. Static user+item bias (shrinkage-regularized)
# MAGIC 3. Temporal bias (binned movie drift + smooth per-user time drift)
# MAGIC 4. Plain ALS on raw ratings
# MAGIC 5. Temporal-bias-adjusted ALS
# MAGIC
# MAGIC (Rung 6, the blend, is a separate later step against saved artifacts.)
# MAGIC
# MAGIC **Data flow:** raw text is converted ONCE to chunked Parquet on DBFS
# MAGIC (Stage 0); every run afterward reads Parquet directly. The v1 pipeline
# MAGIC parsed all ~100M rows into Python lists on the driver and crashed it (OOM,
# MAGIC run 613793594430932, 2026-07-13) -- documented honestly here because it is
# MAGIC exactly the kind of scale lesson this project is about.

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath("__file__")))

from pyspark.sql import functions as F

from parser import convert_to_parquet, parse_netflix_full, to_spark_df, load_probe_pairs
from temporal_bias import compute_global_mean, fit_bias_model, apply_bias
from train_als import fit_als, save_pipeline_artifacts
from evaluate import (split_train_probe, score_constant, score_bias_only,
                      score_als_raw, score_als_with_bias, format_ladder)

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

DATA_PATH = "/dbfs/netflix_prize_data"            # raw text (uploaded via CLI)
PARQUET_PATH = "/dbfs/netflix_prize_data/parquet"  # one-time converted copy
OUTPUT_PATH = "/dbfs/data612_final_project/pipeline_artifacts"

RANK = 20          # Project 5's winning config -- the sweep re-tunes these later
REG_PARAM = 0.1
MAX_ITER = 10
SEED = 45
BIN_DAYS = 30

# Staged execution: comma-separated subset via the "data_files" job parameter
# (e.g. "combined_data_1.txt"). Empty = all four files (full corpus).
try:
    dbutils.widgets.text("data_files", "")
    _files_param = dbutils.widgets.get("data_files").strip()
except NameError:  # running outside Databricks (local test) -- no dbutils
    _files_param = ""
DATA_FILES = [f.strip() for f in _files_param.split(",") if f.strip()] or None
print(f"Data files this run: {DATA_FILES or 'ALL FOUR (full corpus)'}")

pipeline_start = time.time()
ladder = []  # accumulates (rung_name, scores) across the run

# COMMAND ----------

# MAGIC %md ## Stage 0 — One-time Parquet conversion (skipped if already done)
# MAGIC Chunked (5M rows/chunk) so driver memory stays bounded at any corpus size.

# COMMAND ----------

t0 = time.time()
if not os.path.exists(os.path.join(PARQUET_PATH, "part_0000.parquet")):
    total = convert_to_parquet(DATA_PATH, PARQUET_PATH)  # always converts ALL files
    print(f"Converted {total:,} ratings to Parquet in {time.time() - t0:.1f}s")
else:
    print("Parquet already exists -- skipping conversion")

# COMMAND ----------

# MAGIC %md ## Stage 1 — Load ratings from Parquet

# COMMAND ----------

t0 = time.time()
ratings_df = spark.read.parquet(PARQUET_PATH.replace("/dbfs", "dbfs:", 1))
if DATA_FILES:
    # staged run: approximate the file subset by movie-id range (file 1 holds
    # movies 1-4499, file 2 4500-9210, file 3 9211-13367, file 4 13368-17770)
    ranges = {"combined_data_1.txt": (1, 4499), "combined_data_2.txt": (4500, 9210),
              "combined_data_3.txt": (9211, 13367), "combined_data_4.txt": (13368, 17770)}
    conds = None
    for f_ in DATA_FILES:
        lo, hi = ranges[f_]
        c = (F.col("movie_id") >= lo) & (F.col("movie_id") <= hi)
        conds = c if conds is None else (conds | c)
    ratings_df = ratings_df.filter(conds)
ratings_df = ratings_df.repartition(64).cache()
row_count = ratings_df.count()
print(f"Loaded {row_count:,} ratings in {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Stage 2 — Probe split

# COMMAND ----------

t0 = time.time()
probe_pairs_df = to_spark_df(spark, load_probe_pairs(DATA_PATH))
train_df, probe_truth_df = split_train_probe(ratings_df, probe_pairs_df)
train_df = train_df.cache()
probe_truth_df = probe_truth_df.cache()
train_rows = train_df.count()
probe_rows = probe_truth_df.count()
print(f"Train rows: {train_rows:,} | Probe rows: {probe_rows:,} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Rung 1 — Global mean

# COMMAND ----------

t0 = time.time()
global_mean = compute_global_mean(train_df)
ladder.append(("1. global mean", score_constant(probe_truth_df, global_mean)))
print(f"Global mean: {global_mean:.4f} | {ladder[-1][1]} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Rung 2 — Static user+item bias (shrinkage-regularized, no time terms)

# COMMAND ----------

t0 = time.time()
static_bias = fit_bias_model(train_df, global_mean, temporal=False, bin_days=BIN_DAYS)
ladder.append(("2. static user+item bias", score_bias_only(probe_truth_df, static_bias)))
print(f"{ladder[-1][1]} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Rung 3 — Temporal bias (binned movie drift + smooth per-user drift)

# COMMAND ----------

t0 = time.time()
temporal_bias = fit_bias_model(train_df, global_mean, temporal=True, bin_days=BIN_DAYS)
ladder.append(("3. temporal bias", score_bias_only(probe_truth_df, temporal_bias)))
print(f"{ladder[-1][1]} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Rung 4 — Plain ALS on raw ratings (no bias adjustment)

# COMMAND ----------

raw_train = train_df.select("user_id", "movie_id", F.col("rating").cast("float").alias("rating"))
model_raw, fit_secs = fit_als(raw_train, rank=RANK, reg_param=REG_PARAM,
                              max_iter=MAX_ITER, seed=SEED, rating_col="rating")
t0 = time.time()
ladder.append(("4. plain ALS (raw ratings)",
               score_als_raw(model_raw, probe_truth_df, total_probe_rows=probe_rows)))
print(f"fit {fit_secs:.1f}s | eval {time.time() - t0:.1f}s | {ladder[-1][1]}")

# COMMAND ----------

# MAGIC %md ## Rung 5 — Temporal-bias-adjusted ALS

# COMMAND ----------

t0 = time.time()
adjusted = apply_bias(train_df, temporal_bias)
als_train = adjusted.select("user_id", "movie_id", "residual")
model_residual, fit_secs = fit_als(als_train, rank=RANK, reg_param=REG_PARAM,
                                   max_iter=MAX_ITER, seed=SEED, rating_col="residual")
ladder.append(("5. temporal-bias-adjusted ALS",
               score_als_with_bias(model_residual, temporal_bias, probe_truth_df,
                                   total_probe_rows=probe_rows)))
print(f"fit {fit_secs:.1f}s | total rung {time.time() - t0:.1f}s | {ladder[-1][1]}")

# COMMAND ----------

# MAGIC %md ## Ladder summary + save artifacts

# COMMAND ----------

print(format_ladder(ladder))

# COMMAND ----------

t0 = time.time()
save_pipeline_artifacts(OUTPUT_PATH, model_residual, temporal_bias)
print(f"Artifacts saved to {OUTPUT_PATH} in {time.time() - t0:.1f}s")
print(f"\nTOTAL PIPELINE TIME: {time.time() - pipeline_start:.1f}s")
