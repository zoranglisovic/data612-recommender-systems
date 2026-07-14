"""
Generates a qualifying.txt-formatted prediction file -- the exact submission
format the original Netflix Prize competition scored. Netflix never published
the answer key after the competition closed, so this file can't be scored
against anything; producing it demonstrates the pipeline runs end to end in the
real competition's terms (planning document section 2).

Output format (mirrors the input qualifying.txt):
    <movie_id>:
    <prediction>
    <prediction>
    ...
one prediction line per (user, date) row, in the same order as the input file.
"""

import os

from pyspark.sql import functions as F

from parser import load_qualifying_pairs, to_spark_df
from temporal_bias import apply_bias, clip_to_scale


def generate_qualifying_predictions(spark, data_path, als_model, bias_model,
                                    out_path, qualifying_file="qualifying.txt"):
    """Predicts every qualifying pair with the final model (bias + ALS residual)
    and writes the submission-format file to out_path. Rows whose user or movie
    was never seen in training fall back to the bias model alone (which itself
    falls back toward the global mean) -- a prediction file must have a line for
    EVERY input row, so coldStartStrategy='drop' rows get bias-only predictions
    instead of disappearing."""
    qual_pd = load_qualifying_pairs(data_path, filename=qualifying_file)
    # keep the original file order recoverable after Spark shuffles
    qual_pd = qual_pd.reset_index().rename(columns={"index": "row_order"})
    qual_df = to_spark_df(spark, qual_pd)

    with_bias = apply_bias(qual_df, bias_model)
    preds = als_model.transform(with_bias)  # 'prediction' = residual; NaN when unseen
    preds = preds.withColumn(
        "raw_pred",
        F.col("bias_pred") + F.coalesce(
            F.when(F.isnan(F.col("prediction")), None).otherwise(F.col("prediction")),
            F.lit(0.0),
        ),
    )
    preds = clip_to_scale(preds, "raw_pred", "final_pred")

    ordered = (
        preds.select("row_order", "movie_id", "final_pred")
        .orderBy("row_order")
        .toPandas()
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = []
    current_movie = None
    for movie_id, pred in zip(ordered["movie_id"], ordered["final_pred"]):
        if movie_id != current_movie:
            lines.append(f"{movie_id}:")
            current_movie = movie_id
        lines.append(f"{pred:.4f}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(ordered)
