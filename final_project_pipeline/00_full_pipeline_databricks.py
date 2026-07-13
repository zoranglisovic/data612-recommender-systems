# Databricks notebook source
# MAGIC %md
# MAGIC # DATA 612 Final Project — Full Pipeline Timing Test
# MAGIC
# MAGIC Runs the temporal-bias-adjusted ALS pipeline against the **real, full Netflix
# MAGIC Prize corpus** on Databricks, to find out how long each stage actually takes
# MAGIC and whether it completes at all on this cluster -- not a synthetic smoke test.
# MAGIC
# MAGIC **Before running:** update `DATA_PATH` below to wherever the Netflix Prize
# MAGIC data (`combined_data_1-4.txt`, `probe.txt`, `qualifying.txt`, `movie_titles.csv`)
# MAGIC actually lives on this workspace -- a DBFS path, a Unity Catalog Volume, or the
# MAGIC `/mnt/myblob/` mount already configured in this tenant. See `README.md` in this
# MAGIC folder for upload options. This notebook does not upload the data itself.
# MAGIC
# MAGIC Each stage prints its own wall-clock time and row counts, and every
# MAGIC intermediate result is cached with `.count()` immediately after being built --
# MAGIC so a failure or a slow stage shows up exactly where it happens, rather than
# MAGIC being deferred to a later action by Spark's lazy evaluation.

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath("__file__")))

from pyspark.sql import functions as F

from parser import parse_netflix_full, to_spark_df, load_probe_pairs
from temporal_bias import compute_global_mean, compute_residual, add_time_bin
from train_als import fit_als, save_pipeline_artifacts
from evaluate import split_train_probe, global_mean_baseline_rmse, evaluate_als_on_probe

# COMMAND ----------

# MAGIC %md ## Configuration -- edit these two paths before running

# COMMAND ----------

DATA_PATH = "/dbfs/netflix_prize_data"  # uploaded via `databricks fs cp` (CLI profile: data612)
OUTPUT_PATH = "/dbfs/data612_final_project/pipeline_artifacts"

RANK = 20          # Project 5's winning rank -- starting point, not yet re-tuned for full scale
REG_PARAM = 0.1    # same -- plan calls for re-sweeping both on this corpus's own validation slice
MAX_ITER = 10
SEED = 45
BIN_DAYS = 30      # coarse per-movie time-window width -- a tuning decision, not settled

pipeline_start = time.time()

# COMMAND ----------

# MAGIC %md ## Stage 1 — Parse the full corpus (single-threaded Python scan)

# COMMAND ----------

t0 = time.time()
ratings_pd = parse_netflix_full(DATA_PATH)
print(f"Parsed {len(ratings_pd):,} ratings in {time.time() - t0:.1f}s")

# COMMAND ----------

t0 = time.time()
ratings_df = to_spark_df(spark, ratings_pd)
ratings_df = ratings_df.repartition(64)  # 64 partitions for an 8-core node handling ~100M rows
row_count = ratings_df.count()
print(f"Converted to Spark DataFrame ({row_count:,} rows) in {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Stage 2 — Load probe.txt and split train / held-out probe set

# COMMAND ----------

t0 = time.time()
probe_pairs_pd = load_probe_pairs(DATA_PATH)
probe_pairs_df = to_spark_df(spark, probe_pairs_pd)
train_df, probe_truth_df = split_train_probe(ratings_df, probe_pairs_df)

train_rows = train_df.count()
probe_rows = probe_truth_df.count()
print(f"Train rows: {train_rows:,} | Probe rows: {probe_rows:,} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Rung 1 — Global mean baseline (sanity check)

# COMMAND ----------

t0 = time.time()
global_mean = compute_global_mean(train_df)
baseline_rmse = global_mean_baseline_rmse(probe_truth_df, global_mean)
print(f"Global mean: {global_mean:.4f} | Baseline RMSE: {baseline_rmse:.4f} | took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Stage 3 — Temporal bias adjustment (Rungs 2-3 of the ablation ladder)

# COMMAND ----------

t0 = time.time()
adjusted_df, movie_bias_df, user_bias_df = compute_residual(train_df, global_mean, bin_days=BIN_DAYS)
adjusted_rows = adjusted_df.count()
print(f"Temporal bias computed, {adjusted_rows:,} adjusted rows, took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Stage 4 — ALS fit on the bias-adjusted residual (Rung 4-5)

# COMMAND ----------

als_train_df = adjusted_df.select("user_id", "movie_id", "residual")
model, als_fit_seconds = fit_als(
    als_train_df, rank=RANK, reg_param=REG_PARAM, max_iter=MAX_ITER, seed=SEED
)
print(f"ALS fit completed in {als_fit_seconds:.1f}s (rank={RANK}, regParam={REG_PARAM})")

# COMMAND ----------

# MAGIC %md ## Stage 5 — Evaluate on probe.txt (real held-out RMSE)

# COMMAND ----------

t0 = time.time()
probe_truth_binned = add_time_bin(probe_truth_df, bin_days=BIN_DAYS)
results = evaluate_als_on_probe(model, movie_bias_df, user_bias_df, global_mean, probe_truth_binned, bin_days=BIN_DAYS)
print(f"Probe RMSE: {results['rmse']:.4f}")
print(f"Scored {results['scored_rows']:,} / {results['total_probe_rows']:,} probe rows "
      f"({results['drop_rate']:.2%} dropped, coldStartStrategy='drop')")
print(f"Evaluation took {time.time() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## Stage 6 — Save model + bias tables for later reuse (no refitting needed)

# COMMAND ----------

t0 = time.time()
save_pipeline_artifacts(OUTPUT_PATH, model, movie_bias_df, user_bias_df, global_mean)
print(f"Artifacts saved to {OUTPUT_PATH} in {time.time() - t0:.1f}s")

print(f"\nTOTAL PIPELINE TIME: {time.time() - pipeline_start:.1f}s")
