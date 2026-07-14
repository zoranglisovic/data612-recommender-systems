# Databricks notebook source
# MAGIC %md
# MAGIC # DATA 612 Final Project — Bias Tuning, ALS Sweep, Blend (Rung 6)
# MAGIC
# MAGIC Three stages, all tuned on a validation slice carved from training --
# MAGIC probe is touched exactly once, at the very end:
# MAGIC
# MAGIC 1. **Bias λ mini-sweep** (cheap, no ALS): tunes the temporal bias model's
# MAGIC    shrinkage strengths. The full-corpus ladder runs showed untuned temporal
# MAGIC    terms actually hurt vs. static bias (0.9978 vs 0.9843 probe RMSE) -- but
# MAGIC    since the temporal model degenerates to static as λ→∞, tuned λs can at
# MAGIC    worst tie static. This measures what temporal modeling really buys.
# MAGIC 2. **ALS rank × regParam sweep** at full corpus scale on the winning bias
# MAGIC    model's residuals.
# MAGIC 3. **Blend**: best residual-ALS + a plain raw-ratings ALS -- two
# MAGIC    structurally different models (the raw model learns biases implicitly in
# MAGIC    its factors; the residual model gets them explicitly), which gives the
# MAGIC    blend real diversity. Weights from linear regression on validation.
# MAGIC
# MAGIC **Honest deviation from the planning document:** the blend partner is a
# MAGIC second Spark ALS, not the `surprise` SVD baseline the plan named --
# MAGIC `surprise` is single-machine in-memory SGD and cannot scale to ~100M rows.
# MAGIC The blend keeps the plan's idea (differently-shaped models beat either
# MAGIC alone, the Prize-winning ensemble principle) with an implementation that
# MAGIC runs at this scale.

# COMMAND ----------

import json
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
from evaluate import split_train_probe, score_predictions, score_bias_only

# COMMAND ----------

PARQUET_PATH = "dbfs:/netflix_prize_data/parquet"
DATA_PATH = "/dbfs/netflix_prize_data"
# split-mode-specific output so the random-split and time-split sweep results
# both survive for the notebook's side-by-side comparison (set below)

SEED = 45
BIN_DAYS = 30
MAX_ITER = 10
BIAS_GRID = [(lb, ld) for lb in (50.0, 200.0, 1000.0) for ld in (500.0, 5000.0, 50000.0)]
ALS_GRID = [(rank, reg) for rank in (20, 40) for reg in (0.05, 0.1, 0.2)]
VALIDATION_FRACTION = 0.02  # used only by split_mode="random"

# split_mode widget: "time" (default) holds out each user's LAST ratings, the
# way Netflix built probe itself; "random" is the first sweep's uniform sample.
# The first (random-split) sweep picked rank=40/reg=0.05 at 0.7742 validation
# RMSE that transferred to only 0.9453 on probe -- worse than untuned
# rank=20/reg=0.1 (0.9373). Random within-history validation rewards
# overfitting the past; probe is the future. This run tunes on a
# probe-shaped split instead.
try:
    dbutils.widgets.text("split_mode", "time")
    SPLIT_MODE = dbutils.widgets.get("split_mode").strip() or "time"
except NameError:
    SPLIT_MODE = "time"

OUTPUT_PATH = f"/dbfs/data612_final_project/sweep_artifacts_{SPLIT_MODE}"

t_start = time.time()

# COMMAND ----------

# MAGIC %md ## Data: train / validation / probe

# COMMAND ----------

from pyspark.sql import Window

ratings_df = spark.read.parquet(PARQUET_PATH)
probe_pairs_df = to_spark_df(spark, load_probe_pairs(DATA_PATH))
train_full_df, probe_truth_df = split_train_probe(ratings_df, probe_pairs_df)

if SPLIT_MODE == "time":
    # Hold out each user's 2 most recent training ratings (only for users with
    # >= 10 ratings, so sparse users keep their full history in training).
    # ~1M rows -- comparable to probe's 1.4M, and shaped the same way.
    w_user = Window.partitionBy("user_id").orderBy(
        F.col("date").desc(), F.col("movie_id").desc()
    )
    flagged = train_full_df.withColumn("recency_rank", F.row_number().over(w_user))
    flagged = flagged.withColumn(
        "n_user", F.count("*").over(Window.partitionBy("user_id"))
    )
    is_val = (F.col("recency_rank") <= 2) & (F.col("n_user") >= 10)
    val_df = flagged.filter(is_val).drop("recency_rank", "n_user")
    train_df = flagged.filter(~is_val).drop("recency_rank", "n_user")
else:
    train_df, val_df = train_full_df.randomSplit(
        [1.0 - VALIDATION_FRACTION, VALIDATION_FRACTION], seed=SEED
    )

train_df = train_df.repartition(64).cache()
val_df = val_df.cache()
probe_truth_df = probe_truth_df.cache()
print(f"split_mode={SPLIT_MODE} | train: {train_df.count():,} | "
      f"validation: {val_df.count():,} | probe: {probe_truth_df.count():,}")

# COMMAND ----------

# MAGIC %md ## Stage 1 — Bias λ mini-sweep on validation (static baseline included)

# COMMAND ----------

global_mean = compute_global_mean(train_df)

static_bias = fit_bias_model(train_df, global_mean, temporal=False, bin_days=BIN_DAYS)
static_val = score_bias_only(val_df, static_bias)
print(f"static bias (reference)              val RMSE={static_val['rmse']:.4f} MAE={static_val['mae']:.4f}")

bias_sweep = []
best_bias, best_bias_scores, best_bias_cfg = None, None, None
for lam_bin, lam_drift in BIAS_GRID:
    t0 = time.time()
    bm = fit_bias_model(train_df, global_mean, temporal=True, bin_days=BIN_DAYS,
                        lambda_bin=lam_bin, lambda_drift=lam_drift)
    scores = score_bias_only(val_df, bm)
    bias_sweep.append({"lambda_bin": lam_bin, "lambda_drift": lam_drift, **scores})
    print(f"lam_bin={lam_bin:<7} lam_drift={lam_drift:<8} val RMSE={scores['rmse']:.4f} "
          f"MAE={scores['mae']:.4f} ({time.time() - t0:.0f}s)")
    if best_bias_scores is None or scores["rmse"] < best_bias_scores["rmse"]:
        best_bias, best_bias_scores = bm, scores
        best_bias_cfg = {"lambda_bin": lam_bin, "lambda_drift": lam_drift}

temporal_helps = best_bias_scores["rmse"] < static_val["rmse"]
if not temporal_helps:
    # honest fallback: if no temporal config beats static on validation, the
    # static model IS the bias model, and the notebook reports that finding
    best_bias, best_bias_scores, best_bias_cfg = static_bias, static_val, {"static": True}
print(f"\nchosen bias model: {best_bias_cfg} (val RMSE {best_bias_scores['rmse']:.4f}, "
      f"temporal_helps={temporal_helps})")

# COMMAND ----------

# MAGIC %md ## Stage 2 — ALS sweep on the winning bias model's residuals

# COMMAND ----------

adjusted = apply_bias(train_df, best_bias)
als_train = adjusted.select("user_id", "movie_id", "residual").cache()
als_train.count()
val_with_bias = apply_bias(val_df, best_bias).cache()
val_with_bias.count()

als_sweep = []
models = {}
for rank, reg in ALS_GRID:
    model, fit_secs = fit_als(als_train, rank=rank, reg_param=reg,
                              max_iter=MAX_ITER, seed=SEED, rating_col="residual")
    preds = model.transform(val_with_bias)
    preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
    preds = clip_to_scale(preds, "raw_pred", "pred")
    scores = score_predictions(preds, "rating", "pred")
    als_sweep.append({"rank": rank, "regParam": reg, "fit_seconds": fit_secs, **scores})
    models[(rank, reg)] = model
    print(f"rank={rank:<3} regParam={reg:<5} val RMSE={scores['rmse']:.4f} "
          f"MAE={scores['mae']:.4f} fit={fit_secs:.0f}s")

best_als_cfg = min(als_sweep, key=lambda r: r["rmse"])
model_residual = models[(best_als_cfg["rank"], best_als_cfg["regParam"])]
print(f"\nbest residual ALS: rank={best_als_cfg['rank']} reg={best_als_cfg['regParam']} "
      f"val RMSE={best_als_cfg['rmse']:.4f}")

# COMMAND ----------

# MAGIC %md ## Stage 3 — Raw-ratings ALS blend partner (same winning rank/reg)

# COMMAND ----------

raw_train = train_df.select("user_id", "movie_id", F.col("rating").cast("float").alias("rating"))
model_raw, fit_secs = fit_als(raw_train, rank=best_als_cfg["rank"],
                              reg_param=best_als_cfg["regParam"],
                              max_iter=MAX_ITER, seed=SEED, rating_col="rating")
raw_val = model_raw.transform(val_with_bias)
raw_val_scores = score_predictions(clip_to_scale(raw_val, "prediction", "pred"), "rating", "pred")
print(f"raw ALS blend partner: val RMSE={raw_val_scores['rmse']:.4f} (fit {fit_secs:.0f}s)")

# COMMAND ----------

# MAGIC %md ## Blend weights from validation predictions

# COMMAND ----------

pa = model_residual.transform(val_with_bias).withColumnRenamed("prediction", "pred_res")
pb = (model_raw.transform(val_df.select("user_id", "movie_id", "rating"))
      .select("user_id", "movie_id", F.col("prediction").alias("pred_raw")))
both = pa.join(pb, ["user_id", "movie_id"])
both = both.withColumn("full_res", F.col("bias_pred") + F.col("pred_res"))

assembler = VectorAssembler(inputCols=["full_res", "pred_raw"], outputCol="features")
blend_train = assembler.transform(both).select(
    "features", F.col("rating").cast("double").alias("label"))
blend = LinearRegression(featuresCol="features", labelCol="label").fit(blend_train)
w_res, w_raw = list(blend.coefficients)
w0 = blend.intercept
print(f"blend: pred = {w_res:.4f}*residual_model + {w_raw:.4f}*raw_model + {w0:.4f}")

# COMMAND ----------

# MAGIC %md ## Final scoring on probe (first and only probe touch)

# COMMAND ----------

probe_rows = probe_truth_df.count()
probe_bias = apply_bias(probe_truth_df, best_bias)
qa = model_residual.transform(probe_bias).withColumnRenamed("prediction", "pred_res")
qb = (model_raw.transform(probe_truth_df.select("user_id", "movie_id", "rating"))
      .select("user_id", "movie_id", F.col("prediction").alias("pred_raw")))
qboth = qa.join(qb, ["user_id", "movie_id"])
qboth = qboth.withColumn("full_res", F.col("bias_pred") + F.col("pred_res"))

# each model alone on probe, then the blend
single_res = score_predictions(clip_to_scale(qboth, "full_res", "p1"), "rating", "p1")
single_raw = score_predictions(clip_to_scale(qboth, "pred_raw", "p2"), "rating", "p2")
qboth = qboth.withColumn(
    "raw_blend", F.lit(w0) + F.lit(w_res) * F.col("full_res") + F.lit(w_raw) * F.col("pred_raw"))
qboth = clip_to_scale(qboth, "raw_blend", "pred")
blend_scores = score_predictions(qboth, "rating", "pred")

dropped = probe_rows - blend_scores["n_scored"]
print(f"probe — residual ALS alone:  RMSE={single_res['rmse']:.4f} MAE={single_res['mae']:.4f}")
print(f"probe — raw ALS alone:       RMSE={single_raw['rmse']:.4f} MAE={single_raw['mae']:.4f}")
print(f"probe — BLEND (rung 6):      RMSE={blend_scores['rmse']:.4f} MAE={blend_scores['mae']:.4f}")
print(f"scored {blend_scores['n_scored']:,}/{probe_rows:,} ({dropped/probe_rows:.2%} dropped)")

# COMMAND ----------

# MAGIC %md ## Save winning artifacts + results

# COMMAND ----------

save_pipeline_artifacts(OUTPUT_PATH, model_residual, best_bias)
model_raw.write().overwrite().save(OUTPUT_PATH.replace("/dbfs", "dbfs:", 1) + "/als_model_raw")
with open(os.path.join(OUTPUT_PATH, "sweep_results.json"), "w") as f:
    json.dump({
        "static_bias_val": static_val, "bias_sweep": bias_sweep,
        "best_bias_cfg": best_bias_cfg, "temporal_helps": bool(temporal_helps),
        "als_sweep": als_sweep, "best_als_cfg": best_als_cfg,
        "raw_partner_val": raw_val_scores,
        "blend_weights": {"w_res": w_res, "w_raw": w_raw, "intercept": w0},
        "probe": {"residual_alone": single_res, "raw_alone": single_raw,
                  "blend": blend_scores},
    }, f, indent=2, default=float)
print(f"saved to {OUTPUT_PATH}")
print(f"\nTOTAL: {time.time() - t_start:.1f}s")
