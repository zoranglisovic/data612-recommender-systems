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


def save_pipeline_artifacts(output_dir, als_model, movie_bias_df, user_bias_df, global_mean):
    """Persists everything downstream steps need, so RMSE evaluation, precision@k/
    recall@k, and the qualifying.txt prediction pass can all run later against
    saved artifacts instead of re-fitting -- the same "train once, test many times"
    pattern Project 6 used for its deployed model."""
    os.makedirs(output_dir, exist_ok=True)
    als_model.write().overwrite().save(os.path.join(output_dir, "als_model"))
    movie_bias_df.write.mode("overwrite").parquet(os.path.join(output_dir, "movie_bias"))
    user_bias_df.write.mode("overwrite").parquet(os.path.join(output_dir, "user_bias"))
    with open(os.path.join(output_dir, "global_mean.txt"), "w") as f:
        f.write(str(global_mean))


def load_pipeline_artifacts(spark, output_dir):
    from pyspark.ml.recommendation import ALSModel

    als_model = ALSModel.load(os.path.join(output_dir, "als_model"))
    movie_bias_df = spark.read.parquet(os.path.join(output_dir, "movie_bias"))
    user_bias_df = spark.read.parquet(os.path.join(output_dir, "user_bias"))
    with open(os.path.join(output_dir, "global_mean.txt")) as f:
        global_mean = float(f.read())
    return als_model, movie_bias_df, user_bias_df, global_mean
