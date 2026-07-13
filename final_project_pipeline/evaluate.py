"""
Scores RMSE against probe.txt -- the actual Netflix Prize evaluation protocol,
not a fresh random split. probe.txt lists (movie_id, user_id) pairs whose true
rating already lives inside combined_data; those pairs get held out of training,
then scored after fitting.
"""

import math

from pyspark.sql import functions as F

from temporal_bias import add_bias_back


def split_train_probe(full_df, probe_pairs_df):
    """left_anti removes exactly the probe pairs from training; inner join on the
    same key extracts their true ratings for scoring later. Both use the same
    (movie_id, user_id) key so this is a clean partition, not an approximation."""
    train_df = full_df.join(probe_pairs_df, ["movie_id", "user_id"], "left_anti")
    probe_truth_df = full_df.join(probe_pairs_df, ["movie_id", "user_id"], "inner")
    return train_df, probe_truth_df


def rmse(predictions_df, label_col, prediction_col):
    row = predictions_df.agg(
        F.sqrt(F.avg((F.col(label_col) - F.col(prediction_col)) ** 2)).alias("rmse")
    ).first()
    return row["rmse"]


def evaluate_als_on_probe(als_model, movie_bias_df, user_bias_df, global_mean,
                           probe_truth_df, bin_days=30, rating_col="rating"):
    """probe_truth_df must carry movie_id/user_id/rating/date (i.e. it's the output
    of split_train_probe's probe_truth_df, with time_bin added the same way
    training data was binned, so bias lookups line up)."""
    from temporal_bias import add_time_bin

    probe_binned = add_time_bin(probe_truth_df, bin_days=bin_days)
    raw_preds = als_model.transform(probe_binned)  # adds "prediction" = predicted residual
    final_preds = add_bias_back(raw_preds, movie_bias_df, user_bias_df, global_mean)

    # ALS's coldStartStrategy="drop" removes unseen user/movie rows from raw_preds --
    # matches the plan's note that a nonzero drop rate is expected and must be reported,
    # not silently ignored.
    scored_rows = final_preds.count()
    total_probe_rows = probe_truth_df.count()
    dropped = total_probe_rows - scored_rows

    score = rmse(final_preds, rating_col, "final_prediction")
    return {
        "rmse": score,
        "scored_rows": scored_rows,
        "total_probe_rows": total_probe_rows,
        "dropped_rows": dropped,
        "drop_rate": dropped / total_probe_rows if total_probe_rows else 0.0,
    }


def global_mean_baseline_rmse(probe_truth_df, global_mean, rating_col="rating"):
    """Rung 1 of the ablation ladder -- trivial sanity-check baseline."""
    return probe_truth_df.agg(
        F.sqrt(F.avg((F.col(rating_col) - F.lit(global_mean)) ** 2)).alias("rmse")
    ).first()["rmse"]
