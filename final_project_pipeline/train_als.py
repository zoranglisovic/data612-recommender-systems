"""
Fits ALS on the bias-adjusted residual and saves the resulting model + bias
tables to disk, so downstream evaluation/blending doesn't require refitting.

Default hyperparameters (rank=20, regParam=0.1, maxIter=10, seed=45) are
Project 5's winning config on its 2,500-movie/21.9M-rating subset -- a starting
point, not a settled choice. The plan explicitly calls for re-sweeping regParam/
rank on the full corpus's own validation slice, since a value tuned at one
scale/sparsity doesn't reliably transfer (the same lesson Project 5 itself
learned when SVD's reg_all didn't transfer to ALS's regParam untouched).
"""

import os
import time

from pyspark.ml.recommendation import ALS


def fit_als(train_df, rank=20, reg_param=0.1, max_iter=10, seed=45, user_col="user_id",
            item_col="movie_id", rating_col="residual"):
    als = ALS(
        userCol=user_col,
        itemCol=item_col,
        ratingCol=rating_col,
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        coldStartStrategy="drop",
        seed=seed,
    )
    t0 = time.time()
    model = als.fit(train_df)
    fit_seconds = time.time() - t0
    return model, fit_seconds


def _spark_path(path):
    """On Databricks, Python file ops use the /dbfs FUSE mount while Spark and
    MLlib writers resolve plain paths against the default DBFS filesystem --
    passing "/dbfs/foo" to Spark would silently write to dbfs:/dbfs/foo. This
    maps the FUSE form to the explicit dbfs: URI so both sides agree. Local
    paths (tests) pass through unchanged."""
    return "dbfs:" + path[len("/dbfs"):] if path.startswith("/dbfs/") else path


def save_pipeline_artifacts(output_dir, als_model, bias_model):
    """Persists everything downstream steps need, so RMSE evaluation, precision@k/
    recall@k, and the qualifying.txt prediction pass can all run later against
    saved artifacts instead of re-fitting -- the same "train once, test many times"
    pattern Project 6 used for its deployed model.

    output_dir is the FUSE/local form (e.g. /dbfs/... on Databricks)."""
    import json

    os.makedirs(output_dir, exist_ok=True)
    spark_dir = _spark_path(output_dir)
    als_model.write().overwrite().save(os.path.join(spark_dir, "als_model"))
    if bias_model["movie_bias_binned"] is not None:
        bias_model["movie_bias_binned"].write.mode("overwrite").parquet(
            os.path.join(spark_dir, "movie_bias_binned"))
    bias_model["movie_bias_overall"].write.mode("overwrite").parquet(
        os.path.join(spark_dir, "movie_bias_overall"))
    bias_model["user_model"].write.mode("overwrite").parquet(
        os.path.join(spark_dir, "user_model"))
    meta = {k: bias_model[k] for k in ("global_mean", "bin_days", "temporal")}
    with open(os.path.join(output_dir, "bias_meta.json"), "w") as f:
        json.dump(meta, f)


def load_pipeline_artifacts(spark, output_dir):
    import json

    from pyspark.ml.recommendation import ALSModel

    spark_dir = _spark_path(output_dir)
    als_model = ALSModel.load(os.path.join(spark_dir, "als_model"))
    with open(os.path.join(output_dir, "bias_meta.json")) as f:
        meta = json.load(f)
    # os.path.exists sees the FUSE/local form; Spark reads the URI form
    has_binned = os.path.exists(os.path.join(output_dir, "movie_bias_binned"))
    bias_model = {
        "global_mean": meta["global_mean"],
        "bin_days": meta["bin_days"],
        "temporal": meta["temporal"],
        "movie_bias_binned": (spark.read.parquet(os.path.join(spark_dir, "movie_bias_binned"))
                              if has_binned else None),
        "movie_bias_overall": spark.read.parquet(os.path.join(spark_dir, "movie_bias_overall")),
        "user_model": spark.read.parquet(os.path.join(spark_dir, "user_model")),
    }
    return als_model, bias_model
