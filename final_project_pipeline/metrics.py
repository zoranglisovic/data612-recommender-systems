"""
Beyond-accuracy metrics on the probe predictions -- precision@k, recall@k, and
catalog coverage -- carried forward from Projects 4 and 5, as the check on
whether RMSE improvements actually translate into better-looking
recommendations.

Definitions follow the Project 4/5 convention:
- A probe item is *relevant* for a user if its true rating >= threshold (4.0).
- Each user's probe items are ranked by predicted rating; the top k form the
  recommendation list. Only users with at least one relevant probe item and at
  least k probe items are scored (per-user averages over that population).
- Coverage = fraction of the full movie catalog that appears in at least one
  user's top-k list.

All computed with Spark window functions -- no per-user Python loops.
"""

from pyspark.sql import Window
from pyspark.sql import functions as F


def precision_recall_at_k(scored_probe_df, k=10, threshold=4.0,
                          rating_col="rating", pred_col="pred"):
    """scored_probe_df: one row per (user, movie) probe pair with true rating and
    prediction (e.g. the output the evaluate.py scoring functions build).
    Returns dict with precision@k, recall@k, and the user count scored."""
    w = Window.partitionBy("user_id").orderBy(F.col(pred_col).desc())
    ranked = scored_probe_df.withColumn("rank", F.row_number().over(w))
    ranked = ranked.withColumn(
        "relevant", (F.col(rating_col) >= F.lit(threshold)).cast("int")
    )

    per_user = ranked.groupBy("user_id").agg(
        F.sum(F.when(F.col("rank") <= k, F.col("relevant")).otherwise(0)).alias("hits"),
        F.sum("relevant").alias("total_relevant"),
        F.count("*").alias("n_probe_items"),
    )
    scorable = per_user.filter(
        (F.col("total_relevant") > 0) & (F.col("n_probe_items") >= k)
    )
    row = scorable.agg(
        F.avg(F.col("hits") / F.lit(k)).alias("precision_at_k"),
        F.avg(F.col("hits") / F.col("total_relevant")).alias("recall_at_k"),
        F.count("*").alias("n_users"),
    ).first()
    return {
        "k": k,
        "threshold": threshold,
        "precision_at_k": row["precision_at_k"],
        "recall_at_k": row["recall_at_k"],
        "n_users_scored": row["n_users"],
    }


def catalog_coverage(scored_probe_df, catalog_size, k=10, pred_col="pred"):
    """Fraction of the full catalog appearing in at least one user's top-k.
    catalog_size should be the TRUE catalog size (17,770 for the full corpus),
    not just the movies that happen to appear in probe -- coverage against the
    real catalog is the honest number."""
    w = Window.partitionBy("user_id").orderBy(F.col(pred_col).desc())
    ranked = scored_probe_df.withColumn("rank", F.row_number().over(w))
    n_recommended = (
        ranked.filter(F.col("rank") <= k).select("movie_id").distinct().count()
    )
    return {
        "k": k,
        "movies_recommended": n_recommended,
        "catalog_size": catalog_size,
        "coverage": n_recommended / catalog_size if catalog_size else 0.0,
    }
