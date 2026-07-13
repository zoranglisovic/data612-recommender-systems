"""
Temporal bias adjustment: global mean + time-windowed movie bias + user bias,
subtracted before ALS, added back at prediction time.

This is the v1 mechanics already smoke-tested successfully on the 8-core/32GB
Databricks cluster at 200K and 2M synthetic rows (both the DataFrame groupBy/join
step and the RDD-level op the professor flagged as a possible serverless risk --
neither errored). It implements the planning doc's per-movie time-binned drift;
per-user bias here is a plain per-user mean, not yet the continuous smooth-trend
regression the planning doc describes -- that refinement is still to be added
once the pipeline's basic timing/scale is confirmed on the real data.
"""

from pyspark.sql import functions as F


def compute_global_mean(ratings_df, rating_col="rating"):
    return ratings_df.agg(F.avg(rating_col)).first()[0]


def add_time_bin(ratings_df, date_col="date", bin_days=30):
    """Coarse time bins for per-movie drift -- movies accumulate enough ratings
    for binned drift to mean something; 30 days is a starting point, not tuned."""
    return ratings_df.withColumn(
        "time_bin",
        F.floor(F.datediff(F.current_date(), F.col(date_col)) / bin_days),
    )


def compute_movie_bias(ratings_binned_df, global_mean, rating_col="rating"):
    return (
        ratings_binned_df.groupBy("movie_id", "time_bin")
        .agg((F.avg(rating_col) - F.lit(global_mean)).alias("movie_bias"))
    )


def compute_user_bias(ratings_df, global_mean, rating_col="rating"):
    return ratings_df.groupBy("user_id").agg(
        (F.avg(rating_col) - F.lit(global_mean)).alias("user_bias")
    )


def compute_residual(ratings_df, global_mean, bin_days=30, rating_col="rating", date_col="date"):
    """Runs the full bias step end to end: bin, compute movie/user bias, join back,
    subtract to get the residual ALS actually trains on. Returns
    (adjusted_df, movie_bias_df, user_bias_df) -- callers need the bias tables
    to add predictions back later, and to persist them alongside the ALS model
    so evaluation doesn't require refitting bias from scratch."""
    ratings_binned = add_time_bin(ratings_df, date_col=date_col, bin_days=bin_days)
    movie_bias = compute_movie_bias(ratings_binned, global_mean, rating_col=rating_col)
    user_bias = compute_user_bias(ratings_df, global_mean, rating_col=rating_col)

    joined = ratings_binned.join(movie_bias, ["movie_id", "time_bin"], "left").join(
        user_bias, "user_id", "left"
    )
    adjusted = joined.withColumn(
        "residual",
        F.col(rating_col)
        - F.lit(global_mean)
        - F.coalesce(F.col("movie_bias"), F.lit(0.0))
        - F.coalesce(F.col("user_bias"), F.lit(0.0)),
    )
    return adjusted, movie_bias, user_bias


def add_bias_back(predictions_df, movie_bias, user_bias, global_mean, pred_col="prediction"):
    """Adds global mean + movie bias + user bias back onto an ALS prediction on the
    residual, so the result is back in the original 1-5 rating scale. Assumes
    predictions_df already carries movie_id/time_bin/user_id to join bias tables on."""
    joined = predictions_df.join(movie_bias, ["movie_id", "time_bin"], "left").join(
        user_bias, "user_id", "left"
    )
    return joined.withColumn(
        "final_prediction",
        F.col(pred_col)
        + F.lit(global_mean)
        + F.coalesce(F.col("movie_bias"), F.lit(0.0))
        + F.coalesce(F.col("user_bias"), F.lit(0.0)),
    )
