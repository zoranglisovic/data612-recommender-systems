"""
Evaluation against probe.txt -- the actual Netflix Prize protocol -- plus the
ablation-ladder scoring helpers. Every rung reports BOTH RMSE and MAE (the
course's standing two-metric requirement), on the same probe split, so the
ladder isolates what each component contributes.

Ladder (planning document section 4):
  1. global mean                      -> score_constant
  2. static user+item bias            -> fit_bias_model(temporal=False) + score_bias_only
  3. temporal bias                    -> fit_bias_model(temporal=True)  + score_bias_only
  4. plain ALS on raw ratings         -> fit_als(rating_col='rating')   + score_als_raw
  5. temporal-bias-adjusted ALS       -> fit_als on residual            + score_als_with_bias
  6. blend                            -> (separate module, later step)
"""

from pyspark.sql import functions as F

from temporal_bias import apply_bias, clip_to_scale


def split_train_probe(full_df, probe_pairs_df):
    """left_anti removes exactly the probe pairs from training; inner join on the
    same key extracts their true ratings for scoring. A clean partition of the
    corpus, not an approximation."""
    train_df = full_df.join(probe_pairs_df, ["movie_id", "user_id"], "left_anti")
    probe_truth_df = full_df.join(probe_pairs_df, ["movie_id", "user_id"], "inner")
    return train_df, probe_truth_df


def score_predictions(predictions_df, label_col, prediction_col):
    """RMSE + MAE in a single aggregation pass."""
    row = predictions_df.agg(
        F.sqrt(F.avg((F.col(label_col) - F.col(prediction_col)) ** 2)).alias("rmse"),
        F.avg(F.abs(F.col(label_col) - F.col(prediction_col))).alias("mae"),
        F.count(label_col).alias("n"),
    ).first()
    return {"rmse": row["rmse"], "mae": row["mae"], "n_scored": row["n"]}


def score_constant(probe_truth_df, constant, rating_col="rating"):
    """Rung 1: predict the global mean for every probe pair."""
    preds = probe_truth_df.withColumn("pred", F.lit(float(constant)))
    return score_predictions(preds, rating_col, "pred")


def score_bias_only(probe_truth_df, bias_model, rating_col="rating"):
    """Rungs 2-3: the bias model IS the prediction (no factorization). Clipped
    to the 1-5 scale like every final prediction."""
    preds = apply_bias(probe_truth_df, bias_model)
    preds = clip_to_scale(preds, "bias_pred", "pred")
    return score_predictions(preds, rating_col, "pred")


def _drop_stats(scored, total_probe_rows):
    dropped = total_probe_rows - scored["n_scored"]
    scored["total_probe_rows"] = total_probe_rows
    scored["dropped_rows"] = dropped
    scored["drop_rate"] = dropped / total_probe_rows if total_probe_rows else 0.0
    return scored


def score_als_raw(als_model, probe_truth_df, total_probe_rows=None, rating_col="rating"):
    """Rung 4: ALS trained directly on raw ratings; its transform output IS the
    prediction. coldStartStrategy='drop' removes unseen users/movies -- the drop
    rate is part of the result, reported, never hidden."""
    preds = als_model.transform(probe_truth_df)
    preds = clip_to_scale(preds, "prediction", "pred")
    scored = score_predictions(preds, rating_col, "pred")
    total = total_probe_rows if total_probe_rows is not None else probe_truth_df.count()
    return _drop_stats(scored, total)


def score_als_with_bias(als_model, bias_model, probe_truth_df, total_probe_rows=None,
                        rating_col="rating"):
    """Rung 5: final prediction = bias model + ALS's predicted residual."""
    with_bias = apply_bias(probe_truth_df, bias_model)
    preds = als_model.transform(with_bias)  # adds "prediction" = predicted residual
    preds = preds.withColumn("raw_pred", F.col("bias_pred") + F.col("prediction"))
    preds = clip_to_scale(preds, "raw_pred", "pred")
    scored = score_predictions(preds, rating_col, "pred")
    total = total_probe_rows if total_probe_rows is not None else probe_truth_df.count()
    return _drop_stats(scored, total)


def format_ladder(results):
    """results: list of (rung_name, scores_dict). Returns an aligned text table
    for the notebook's printed output."""
    lines = [f"{'Rung':<38} {'RMSE':>8} {'MAE':>8} {'scored':>12} {'dropped':>9}"]
    for name, s in results:
        dropped = s.get("drop_rate")
        drop_txt = f"{dropped:.2%}" if dropped else "-"
        lines.append(
            f"{name:<38} {s['rmse']:>8.4f} {s['mae']:>8.4f} {s['n_scored']:>12,} {drop_txt:>9}"
        )
    return "\n".join(lines)
