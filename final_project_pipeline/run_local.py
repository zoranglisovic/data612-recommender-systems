"""
Local runner: the exact same pipeline modules the Databricks notebooks use,
pointed at the local copy of the Netflix Prize data and run on local Spark --
the Project 5 pattern (local[*], explicit driver memory), scaled up for the
full corpus on a 12-core / 96GB M2 Max.

Why this exists: local iteration is much faster than the quota-capped Azure
node (P5's ALS fit: 21.6s locally vs ~360s for the same scale on the
Standard_D8ds_v4), so the pipeline gets developed and verified here, then the
same unchanged modules run on Databricks for the platform story. Same code both
places = nothing to re-validate on transfer.

Usage (from databricks_pipeline/, in the data612 conda env):
    python run_local.py convert                 # one-time text -> parquet
    python run_local.py ladder [--files combined_data_1.txt]
    python run_local.py sweep  [--split time|random]
    python run_local.py metrics                 # against saved sweep artifacts
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.expanduser(
    "~/Documents/School/CUNY SPS Master/DATA612 Recommender Systems/Resources/Netflix Prize data"
)
LOCAL_BASE = os.path.expanduser(
    "~/Documents/School/CUNY SPS Master/DATA612 Recommender Systems/Final Project/data_local"
)
PARQUET_PATH = os.path.join(LOCAL_BASE, "parquet")

SEED = 45
BIN_DAYS = 30
MAX_ITER = 10
BIAS_GRID = [(lb, ld) for lb in (50.0, 200.0, 1000.0) for ld in (500.0, 5000.0, 50000.0)]
ALS_GRID = [(rank, reg) for rank in (20, 40) for reg in (0.05, 0.1, 0.2)]


def build_spark():
    """P5's local-session pattern: local[*], quiet stderr during JVM startup,
    Arrow on. Driver memory sized for the full corpus on a 96GB machine."""
    from pyspark.sql import SparkSession

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    saved = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        spark = (
            SparkSession.builder.appName("DATA612_FinalProject_Local")
            .master("local[*]")
            .config("spark.driver.memory", "48g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.local.dir", os.path.join(LOCAL_BASE, "spark_tmp"))
            .getOrCreate()
        )
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull_fd)
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark, files=None):
    from pyspark.sql import functions as F

    from parser import to_spark_df, load_probe_pairs
    from evaluate import split_train_probe

    ratings_df = spark.read.parquet(PARQUET_PATH)
    if files:
        ranges = {"combined_data_1.txt": (1, 4499), "combined_data_2.txt": (4500, 9210),
                  "combined_data_3.txt": (9211, 13367), "combined_data_4.txt": (13368, 17770)}
        conds = None
        for f_ in files:
            lo, hi = ranges[f_]
            c = (F.col("movie_id") >= lo) & (F.col("movie_id") <= hi)
            conds = c if conds is None else (conds | c)
        ratings_df = ratings_df.filter(conds)
    probe_pairs_df = to_spark_df(spark, load_probe_pairs(DATA_PATH))
    train_df, probe_df = split_train_probe(ratings_df, probe_pairs_df)
    return train_df, probe_df


def cmd_convert(_args):
    from parser import convert_to_parquet

    os.makedirs(PARQUET_PATH, exist_ok=True)
    if os.path.exists(os.path.join(PARQUET_PATH, "part_0000.parquet")):
        print("parquet already exists -- delete the folder to reconvert")
        return
    t0 = time.time()
    total = convert_to_parquet(DATA_PATH, PARQUET_PATH)
    print(f"converted {total:,} ratings in {time.time() - t0:.1f}s")


def cmd_ladder(args):
    from pyspark.sql import functions as F

    from temporal_bias import compute_global_mean, fit_bias_model, apply_bias
    from train_als import fit_als, save_pipeline_artifacts
    from evaluate import (score_constant, score_bias_only, score_als_raw,
                          score_als_with_bias, format_ladder)

    spark = build_spark()
    t_start = time.time()
    train_df, probe_df = load_data(spark, files=args.files)
    train_df = train_df.repartition(48).cache()
    probe_df = probe_df.cache()
    print(f"train: {train_df.count():,} | probe: {probe_df.count():,}")
    probe_rows = probe_df.count()

    ladder = []
    gm = compute_global_mean(train_df)
    ladder.append(("1. global mean", score_constant(probe_df, gm)))
    print(ladder[-1])

    static_bias = fit_bias_model(train_df, gm, temporal=False, bin_days=BIN_DAYS)
    ladder.append(("2. static user+item bias", score_bias_only(probe_df, static_bias)))
    print(ladder[-1])

    temporal_bias = fit_bias_model(train_df, gm, temporal=True, bin_days=BIN_DAYS)
    ladder.append(("3. temporal bias", score_bias_only(probe_df, temporal_bias)))
    print(ladder[-1])

    raw_train = train_df.select("user_id", "movie_id", F.col("rating").cast("float").alias("rating"))
    m_raw, fs = fit_als(raw_train, rank=20, reg_param=0.1, max_iter=MAX_ITER, seed=SEED,
                        rating_col="rating")
    ladder.append(("4. plain ALS (raw ratings)", score_als_raw(m_raw, probe_df, probe_rows)))
    print(f"fit {fs:.1f}s", ladder[-1])

    adjusted = apply_bias(train_df, temporal_bias)
    m_res, fs = fit_als(adjusted.select("user_id", "movie_id", "residual"),
                        rank=20, reg_param=0.1, max_iter=MAX_ITER, seed=SEED,
                        rating_col="residual")
    ladder.append(("5. temporal-bias-adjusted ALS",
                   score_als_with_bias(m_res, temporal_bias, probe_df, probe_rows)))
    print(f"fit {fs:.1f}s", ladder[-1])

    print()
    print(format_ladder(ladder))
    out = os.path.join(LOCAL_BASE, "ladder_artifacts")
    save_pipeline_artifacts(out, m_res, temporal_bias)
    print(f"artifacts saved to {out}")
    print(f"TOTAL: {time.time() - t_start:.1f}s")
    spark.stop()


def cmd_sweep(args):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    from temporal_bias import compute_global_mean, fit_bias_model, apply_bias, clip_to_scale
    from train_als import fit_als, save_pipeline_artifacts
    from evaluate import score_predictions, score_bias_only, score_als_with_bias

    spark = build_spark()
    t_start = time.time()
    train_full_df, probe_df = load_data(spark, files=args.files)

    if args.split == "time":
        w_user = Window.partitionBy("user_id").orderBy(F.col("date").desc(), F.col("movie_id").desc())
        flagged = train_full_df.withColumn("recency_rank", F.row_number().over(w_user))
        flagged = flagged.withColumn("n_user", F.count("*").over(Window.partitionBy("user_id")))
        is_val = (F.col("recency_rank") <= 2) & (F.col("n_user") >= 10)
        val_df = flagged.filter(is_val).drop("recency_rank", "n_user")
        train_df = flagged.filter(~is_val).drop("recency_rank", "n_user")
    else:
        train_df, val_df = train_full_df.randomSplit([0.98, 0.02], seed=SEED)

    train_df = train_df.repartition(48).cache()
    val_df = val_df.cache()
    probe_df = probe_df.cache()
    print(f"split={args.split} | train: {train_df.count():,} | val: {val_df.count():,} | "
          f"probe: {probe_df.count():,}")

    gm = compute_global_mean(train_df)
    static_bias = fit_bias_model(train_df, gm, temporal=False, bin_days=BIN_DAYS)
    static_val = score_bias_only(val_df, static_bias)
    print(f"static reference: val RMSE {static_val['rmse']:.4f}")

    bias_sweep, best_bias, best_scores, best_cfg = [], None, None, None
    for lam_bin, lam_drift in BIAS_GRID:
        t0 = time.time()
        bm = fit_bias_model(train_df, gm, temporal=True, bin_days=BIN_DAYS,
                            lambda_bin=lam_bin, lambda_drift=lam_drift)
        s = score_bias_only(val_df, bm)
        bias_sweep.append({"lambda_bin": lam_bin, "lambda_drift": lam_drift, **s})
        print(f"lam_bin={lam_bin:<7} lam_drift={lam_drift:<8} val RMSE {s['rmse']:.4f} "
              f"({time.time() - t0:.0f}s)")
        if best_scores is None or s["rmse"] < best_scores["rmse"]:
            best_bias, best_scores = bm, s
            best_cfg = {"lambda_bin": lam_bin, "lambda_drift": lam_drift}
    temporal_helps = best_scores["rmse"] < static_val["rmse"]
    if not temporal_helps:
        best_bias, best_scores, best_cfg = static_bias, static_val, {"static": True}
    print(f"chosen bias: {best_cfg} (temporal_helps={temporal_helps})")

    adjusted = apply_bias(train_df, best_bias)
    als_train = adjusted.select("user_id", "movie_id", "residual").cache()
    als_train.count()
    val_with_bias = apply_bias(val_df, best_bias).cache()
    val_with_bias.count()

    als_sweep, models = [], {}
    for rank, reg in ALS_GRID:
        model, fs = fit_als(als_train, rank=rank, reg_param=reg, max_iter=MAX_ITER,
                            seed=SEED, rating_col="residual")
        preds = model.transform(val_with_bias)
        preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
        preds = clip_to_scale(preds, "raw_pred", "pred")
        s = score_predictions(preds, "rating", "pred")
        als_sweep.append({"rank": rank, "regParam": reg, "fit_seconds": fs, **s})
        models[(rank, reg)] = model
        print(f"rank={rank:<3} reg={reg:<5} val RMSE {s['rmse']:.4f} fit {fs:.0f}s")
    best_als = min(als_sweep, key=lambda r: r["rmse"])
    model_res = models[(best_als["rank"], best_als["regParam"])]
    print(f"best ALS: rank={best_als['rank']} reg={best_als['regParam']}")

    raw_train = train_df.select("user_id", "movie_id", F.col("rating").cast("float").alias("rating"))
    model_raw, fs = fit_als(raw_train, rank=best_als["rank"], reg_param=best_als["regParam"],
                            max_iter=MAX_ITER, seed=SEED, rating_col="rating")
    print(f"raw partner fit {fs:.0f}s")

    pa = model_res.transform(val_with_bias).withColumnRenamed("prediction", "pred_res")
    pb = (model_raw.transform(val_df.select("user_id", "movie_id", "rating"))
          .select("user_id", "movie_id", F.col("prediction").alias("pred_raw")))
    both = pa.join(pb, ["user_id", "movie_id"])
    both = both.withColumn("full_res", F.col("bias_pred") + F.col("pred_res"))
    assembler = VectorAssembler(inputCols=["full_res", "pred_raw"], outputCol="features")
    bt = assembler.transform(both).select("features", F.col("rating").cast("double").alias("label"))
    blend = LinearRegression(featuresCol="features", labelCol="label").fit(bt)
    w_res, w_raw = list(blend.coefficients)
    w0 = blend.intercept
    print(f"blend: {w_res:.4f}*res + {w_raw:.4f}*raw + {w0:.4f}")

    probe_rows = probe_df.count()
    probe_bias = apply_bias(probe_df, best_bias)
    qa = model_res.transform(probe_bias).withColumnRenamed("prediction", "pred_res")
    qb = (model_raw.transform(probe_df.select("user_id", "movie_id", "rating"))
          .select("user_id", "movie_id", F.col("prediction").alias("pred_raw")))
    qboth = qa.join(qb, ["user_id", "movie_id"])
    qboth = qboth.withColumn("full_res", F.col("bias_pred") + F.col("pred_res"))
    single_res = score_predictions(clip_to_scale(qboth, "full_res", "p1"), "rating", "p1")
    single_raw = score_predictions(clip_to_scale(qboth, "pred_raw", "p2"), "rating", "p2")
    qboth = qboth.withColumn("raw_blend",
                             F.lit(w0) + F.lit(w_res) * F.col("full_res") + F.lit(w_raw) * F.col("pred_raw"))
    qboth = clip_to_scale(qboth, "raw_blend", "pred")
    blend_scores = score_predictions(qboth, "rating", "pred")
    print(f"probe residual alone: RMSE {single_res['rmse']:.4f} MAE {single_res['mae']:.4f}")
    print(f"probe raw alone:      RMSE {single_raw['rmse']:.4f} MAE {single_raw['mae']:.4f}")
    print(f"probe BLEND:          RMSE {blend_scores['rmse']:.4f} MAE {blend_scores['mae']:.4f}")

    out = os.path.join(LOCAL_BASE, f"sweep_artifacts_{args.split}")
    save_pipeline_artifacts(out, model_res, best_bias)
    model_raw.write().overwrite().save(os.path.join(out, "als_model_raw"))
    with open(os.path.join(out, "sweep_results.json"), "w") as f:
        json.dump({"static_bias_val": static_val, "bias_sweep": bias_sweep,
                   "best_bias_cfg": best_cfg, "temporal_helps": bool(temporal_helps),
                   "als_sweep": als_sweep, "best_als_cfg": best_als,
                   "blend_weights": {"w_res": w_res, "w_raw": w_raw, "intercept": w0},
                   "probe": {"residual_alone": single_res, "raw_alone": single_raw,
                             "blend": blend_scores}}, f, indent=2, default=float)
    print(f"saved to {out}")
    print(f"TOTAL: {time.time() - t_start:.1f}s")
    spark.stop()


def cmd_metrics(args):
    from pyspark.sql import functions as F

    from temporal_bias import apply_bias, clip_to_scale
    from train_als import load_pipeline_artifacts
    from evaluate import score_predictions
    from metrics import precision_recall_at_k, catalog_coverage
    from qualifying_writer import generate_qualifying_predictions

    spark = build_spark()
    artifacts = os.path.join(LOCAL_BASE, f"sweep_artifacts_{args.split}")
    als_model, bias_model = load_pipeline_artifacts(spark, artifacts)
    _, probe_df = load_data(spark)
    probe_df = probe_df.cache()

    probe_bias = apply_bias(probe_df, bias_model)
    preds = als_model.transform(probe_bias)
    preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
    preds = clip_to_scale(preds, "raw_pred", "pred").cache()
    s = score_predictions(preds, "rating", "pred")
    print(f"probe RMSE {s['rmse']:.4f} MAE {s['mae']:.4f}")
    for k in (5, 10):
        pr = precision_recall_at_k(preds, k=k, threshold=4.0)
        print(f"k={k}: precision {pr['precision_at_k']:.4f} recall {pr['recall_at_k']:.4f} "
              f"(n_users {pr['n_users_scored']:,})")
    cov = catalog_coverage(preds, catalog_size=17_770, k=10)
    print(f"coverage@10: {cov['movies_recommended']:,}/17,770 = {cov['coverage']:.2%}")

    out_file = os.path.join(LOCAL_BASE, "qualifying_predictions.txt")
    n = generate_qualifying_predictions(spark, DATA_PATH, als_model, bias_model, out_file)
    print(f"wrote {n:,} qualifying predictions to {out_file}")
    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert")
    p_ladder = sub.add_parser("ladder")
    p_ladder.add_argument("--files", nargs="*", default=None)
    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--split", choices=["time", "random"], default="time")
    p_sweep.add_argument("--files", nargs="*", default=None)
    p_metrics = sub.add_parser("metrics")
    p_metrics.add_argument("--split", choices=["time", "random"], default="time")
    args = ap.parse_args()
    {"convert": cmd_convert, "ladder": cmd_ladder,
     "sweep": cmd_sweep, "metrics": cmd_metrics}[args.cmd](args)
