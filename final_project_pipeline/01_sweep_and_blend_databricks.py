# Databricks notebook source
# MAGIC %md
# MAGIC # DATA 612 Final Project — Sweep + Blend (Rung 6)
# MAGIC
# MAGIC Runs the rank/regParam sweep **at full corpus scale** on a dedicated
# MAGIC validation slice (never probe), then blends the two best configurations
# MAGIC from different rank families with a linear regression fit on the same
# MAGIC validation slice, and finally scores the blend on probe -- the only time
# MAGIC probe is touched in this notebook.
# MAGIC
# MAGIC **Honest deviation from the planning document:** the blend partner is a
# MAGIC second Spark ALS configuration, not the `surprise` SVD baseline the plan
# MAGIC named. `surprise` cannot scale to ~100M rows (it is single-machine,
# MAGIC in-memory SGD) and does not install cleanly on the Databricks ML runtime.
# MAGIC The blend keeps the plan's actual idea -- two differently-shaped models
# MAGIC beat either alone, the Prize-winning ensemble principle -- while swapping
# MAGIC the second model's implementation for one that runs at this scale.
# MAGIC
# MAGIC Expect this notebook to be the expensive one: 6 sweep fits + 1 refit at
# MAGIC full scale, all submitted as one job so the cluster stays warm throughout.

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath("__file__")))

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F

from parser import to_spark_df, load_probe_pairs
from temporal_bias import compute_global_mean, fit_bias_model, apply_bias, clip_to_scale
from train_als import fit_als, save_pipeline_artifacts
from evaluate import split_train_probe, score_predictions, score_als_with_bias

# COMMAND ----------

PARQUET_PATH = "dbfs:/netflix_prize_data/parquet"
DATA_PATH = "/dbfs/netflix_prize_data"
OUTPUT_PATH = "/dbfs/data612_final_project/sweep_artifacts"

SEED = 45
BIN_DAYS = 30
MAX_ITER = 10
SWEEP_GRID = [(rank, reg) for rank in (20, 40) for reg in (0.05, 0.1, 0.2)]
VALIDATION_FRACTION = 0.02  # ~2M rows at full scale -- plenty for tuning + blend weights

t_start = time.time()

# COMMAND ----------

# MAGIC %md ## Data: train / validation / probe

# COMMAND ----------

ratings_df = spark.read.parquet(PARQUET_PATH)
probe_pairs_df = to_spark_df(spark, load_probe_pairs(DATA_PATH))
train_full_df, probe_truth_df = split_train_probe(ratings_df, probe_pairs_df)

# validation slice carved from training -- probe stays untouched until the very end
train_df, val_df = train_full_df.randomSplit(
    [1.0 - VALIDATION_FRACTION, VALIDATION_FRACTION], seed=SEED
)
train_df = train_df.repartition(64).cache()
val_df = val_df.cache()
probe_truth_df = probe_truth_df.cache()
print(f"train: {train_df.count():,} | validation: {val_df.count():,} | probe: {probe_truth_df.count():,}")

# COMMAND ----------

# MAGIC %md ## Bias model (fit once on the sweep's training split)

# COMMAND ----------

t0 = time.time()
global_mean = compute_global_mean(train_df)
bias_model = fit_bias_model(train_df, global_mean, temporal=True, bin_days=BIN_DAYS)
adjusted = apply_bias(train_df, bias_model)
als_train = adjusted.select("user_id", "movie_id", "residual").cache()
als_train.count()  # materialize once; every sweep fit reuses this
print(f"bias model + residuals ready in {time.time() - t0:.1f}s")

# validation with bias columns precomputed (reused for every config's scoring)
val_with_bias = apply_bias(val_df, bias_model).cache()
val_with_bias.count()

# COMMAND ----------

# MAGIC %md ## Sweep: rank x regParam on the validation slice

# COMMAND ----------

sweep_results = []
models = {}
for rank, reg in SWEEP_GRID:
    model, fit_secs = fit_als(als_train, rank=rank, reg_param=reg,
                              max_iter=MAX_ITER, seed=SEED, rating_col="residual")
    preds = model.transform(val_with_bias)
    preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
    preds = clip_to_scale(preds, "raw_pred", "pred")
    scores = score_predictions(preds, "rating", "pred")
    sweep_results.append({"rank": rank, "regParam": reg, "fit_seconds": fit_secs, **scores})
    models[(rank, reg)] = model
    print(f"rank={rank:<3} regParam={reg:<5} RMSE={scores['rmse']:.4f} "
          f"MAE={scores['mae']:.4f} fit={fit_secs:.1f}s")

# COMMAND ----------

# MAGIC %md ## Pick blend partners: best config from each rank family

# COMMAND ----------

by_rank = {}
for r in sweep_results:
    best = by_rank.get(r["rank"])
    if best is None or r["rmse"] < best["rmse"]:
        by_rank[r["rank"]] = r
partners = sorted(by_rank.values(), key=lambda r: r["rmse"])
best_cfg, second_cfg = partners[0], partners[1]
print(f"best:   rank={best_cfg['rank']} reg={best_cfg['regParam']} RMSE={best_cfg['rmse']:.4f}")
print(f"second: rank={second_cfg['rank']} reg={second_cfg['regParam']} RMSE={second_cfg['rmse']:.4f}")

model_a = models[(best_cfg["rank"], best_cfg["regParam"])]
model_b = models[(second_cfg["rank"], second_cfg["regParam"])]

# COMMAND ----------

# MAGIC %md ## Blend weights: linear regression on validation predictions

# COMMAND ----------

t0 = time.time()
pa = model_a.transform(val_with_bias).withColumnRenamed("prediction", "pred_a")
pb = (model_b.transform(val_with_bias)
      .select("user_id", "movie_id", F.col("prediction").alias("pred_b")))
both = pa.join(pb, ["user_id", "movie_id"])
both = both.withColumn("full_a", F.col("bias_pred") + F.col("pred_a"))
both = both.withColumn("full_b", F.col("bias_pred") + F.col("pred_b"))

assembler = VectorAssembler(inputCols=["full_a", "full_b"], outputCol="features")
blend_train = assembler.transform(both).select("features", F.col("rating").cast("double").alias("label"))
lr = LinearRegression(featuresCol="features", labelCol="label")
blend = lr.fit(blend_train)
w = list(blend.coefficients) + [blend.intercept]
print(f"blend: pred = {w[0]:.4f}*model_a + {w[1]:.4f}*model_b + {w[2]:.4f}  ({time.time() - t0:.1f}s)")

# COMMAND ----------

# MAGIC %md ## Final scoring on probe (first and only probe touch in this notebook)

# COMMAND ----------

probe_rows = probe_truth_df.count()
probe_bias = apply_bias(probe_truth_df, bias_model)
qa = model_a.transform(probe_bias).withColumnRenamed("prediction", "pred_a")
qb = (model_b.transform(probe_bias)
      .select("user_id", "movie_id", F.col("prediction").alias("pred_b")))
qboth = qa.join(qb, ["user_id", "movie_id"])
qboth = qboth.withColumn(
    "raw_blend",
    F.lit(w[2]) + F.lit(w[0]) * (F.col("bias_pred") + F.col("pred_a"))
                + F.lit(w[1]) * (F.col("bias_pred") + F.col("pred_b")),
)
qboth = clip_to_scale(qboth, "raw_blend", "pred")
blend_scores = score_predictions(qboth, "rating", "pred")
dropped = probe_rows - blend_scores["n_scored"]
print(f"RUNG 6 (blend) — probe RMSE: {blend_scores['rmse']:.4f}  MAE: {blend_scores['mae']:.4f}")
print(f"scored {blend_scores['n_scored']:,}/{probe_rows:,} ({dropped/probe_rows:.2%} dropped)")

# best single model on probe, for the ladder comparison
single_scores = score_als_with_bias(model_a, bias_model, probe_truth_df, probe_rows)
print(f"best single config on probe — RMSE: {single_scores['rmse']:.4f}  MAE: {single_scores['mae']:.4f}")

# COMMAND ----------

# MAGIC %md ## Save winning artifacts

# COMMAND ----------

import json

save_pipeline_artifacts(OUTPUT_PATH, model_a, bias_model)
with open(os.path.join(OUTPUT_PATH, "sweep_results.json"), "w") as f:
    json.dump({
        "sweep": sweep_results,
        "best": best_cfg, "second": second_cfg,
        "blend_weights": {"w_a": w[0], "w_b": w[1], "intercept": w[2]},
        "probe_blend": blend_scores, "probe_best_single": single_scores,
    }, f, indent=2, default=float)
model_b.write().overwrite().save(OUTPUT_PATH.replace("/dbfs", "dbfs:", 1) + "/als_model_b")
print(f"saved to {OUTPUT_PATH}")
print(f"\nTOTAL: {time.time() - t_start:.1f}s")
