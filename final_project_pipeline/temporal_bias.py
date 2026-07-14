"""
Temporal bias model v2: shrinkage-regularized biases + smooth per-user time drift.

v1 (plain per-group means) overfit sparse groups badly -- a user with one rating
got a bias that memorized it, and a (movie, time_bin) cell with two ratings got
a "drift" that was mostly noise. That showed up directly in the staged 24M-row
run's probe RMSE (1.0455). v2 applies the standard baseline-predictor fix from
Project 1: shrink each bias toward zero by a regularization count term,
bias = sum(deviation) / (n + lambda), so sparse groups contribute little and
dense groups keep their signal.

v2 also replaces the plain-mean user bias with what the planning document
actually promised: a smooth per-user time drift. Each user's deviations are fit
with a tiny per-user linear model in time,

    dev ~= a_u + b_u * (t - tbar_u)

where a_u is the user's shrunk mean deviation, b_u a shrunk slope in
deviation-per-day, and tbar_u the user's mean rating date. All three come from
one groupBy pass of sufficient statistics (n, sum_dev, sum_t, sum_t2, sum_t_dev)
-- no per-user loops, fully Spark-parallel.

Movie drift stays binned (movies have enough ratings per window for bins to
mean something), with a two-level fallback at prediction time: binned bias
first, the movie's overall shrunk bias if the exact bin was never seen in
training, zero if the movie itself is new.
"""

from pyspark.sql import functions as F

# Reference date for converting dates to integer day offsets. Any fixed date
# before the corpus start (1999-11-11) works; what matters is consistency
# between fit and predict.
EPOCH = "1999-01-01"

# Shrinkage strengths. These are in "equivalent ratings" units: a group needs
# on the order of lambda ratings before its bias is trusted at half strength.
# Starting points informed by the Netflix Prize literature's typical 20-30 for
# bias terms; candidates for the validation sweep later, not settled values.
LAMBDA_MOVIE = 25.0
LAMBDA_USER = 10.0
LAMBDA_DRIFT = 500.0  # slopes need far more evidence than intercepts


def _add_day_number(df, date_col="date"):
    return df.withColumn("day_num", F.datediff(F.col(date_col), F.lit(EPOCH)))


def add_time_bin(df, date_col="date", bin_days=30):
    """Movie-drift time bins, counted from a fixed epoch (not current_date, which
    would silently shift bin boundaries between runs on different days)."""
    df = _add_day_number(df, date_col=date_col)
    return df.withColumn("time_bin", F.floor(F.col("day_num") / bin_days))


def compute_global_mean(ratings_df, rating_col="rating"):
    return ratings_df.agg(F.avg(rating_col)).first()[0]


def fit_bias_model(train_df, global_mean, bin_days=30, temporal=True,
                   rating_col="rating", date_col="date",
                   lambda_movie=LAMBDA_MOVIE, lambda_user=LAMBDA_USER,
                   lambda_drift=LAMBDA_DRIFT):
    """Fits the full bias model on training data. Returns a dict of small
    DataFrames (the "model") that downstream steps join against:

      movie_bias_binned : (movie_id, time_bin, movie_bias)     -- temporal only
      movie_bias_overall: (movie_id, movie_bias_overall)       -- always
      user_model        : (user_id, user_a, user_b, user_tbar) -- b=0 if not temporal

    With temporal=False this degenerates to the classic static user+item bias
    predictor (ablation rung 2); with temporal=True it is rung 3's model.
    """
    df = add_time_bin(train_df, date_col=date_col, bin_days=bin_days)
    df = df.withColumn("dev", F.col(rating_col) - F.lit(global_mean))

    # --- movie side ---
    movie_bias_overall = df.groupBy("movie_id").agg(
        (F.sum("dev") / (F.count("dev") + F.lit(lambda_movie))).alias("movie_bias_overall")
    )
    if temporal:
        movie_bias_binned = df.groupBy("movie_id", "time_bin").agg(
            (F.sum("dev") / (F.count("dev") + F.lit(lambda_movie))).alias("movie_bias")
        )
    else:
        movie_bias_binned = None

    # --- user side: deviations AFTER the movie bias, so user terms explain what
    # movie terms haven't already ---
    with_movie = df.join(movie_bias_overall, "movie_id", "left")
    if temporal and movie_bias_binned is not None:
        with_movie = with_movie.join(movie_bias_binned, ["movie_id", "time_bin"], "left")
        with_movie = with_movie.withColumn(
            "movie_term",
            F.coalesce(F.col("movie_bias"), F.col("movie_bias_overall"), F.lit(0.0)),
        )
    else:
        with_movie = with_movie.withColumn(
            "movie_term", F.coalesce(F.col("movie_bias_overall"), F.lit(0.0))
        )
    with_movie = with_movie.withColumn("user_dev", F.col("dev") - F.col("movie_term"))

    stats = with_movie.groupBy("user_id").agg(
        F.count("user_dev").alias("n"),
        F.sum("user_dev").alias("sum_dev"),
        F.sum("day_num").alias("sum_t"),
        F.sum(F.col("day_num") * F.col("day_num")).alias("sum_t2"),
        F.sum(F.col("day_num") * F.col("user_dev")).alias("sum_t_dev"),
    )
    user_model = stats.withColumn(
        "user_a", F.col("sum_dev") / (F.col("n") + F.lit(lambda_user))
    ).withColumn("user_tbar", F.col("sum_t") / F.col("n"))

    if temporal:
        # Sxx = sum_t2 - sum_t^2/n ; Sxy = sum_t_dev - sum_t*sum_dev/n
        user_model = user_model.withColumn(
            "sxx", F.col("sum_t2") - F.col("sum_t") * F.col("sum_t") / F.col("n")
        ).withColumn(
            "sxy", F.col("sum_t_dev") - F.col("sum_t") * F.col("sum_dev") / F.col("n")
        ).withColumn(
            "user_b", F.col("sxy") / (F.col("sxx") + F.lit(lambda_drift))
        )
    else:
        user_model = user_model.withColumn("user_b", F.lit(0.0))

    user_model = user_model.select("user_id", "user_a", "user_b", "user_tbar")

    return {
        "global_mean": global_mean,
        "bin_days": bin_days,
        "temporal": temporal,
        "movie_bias_binned": movie_bias_binned,
        "movie_bias_overall": movie_bias_overall,
        "user_model": user_model,
    }


def apply_bias(df, bias_model, date_col="date"):
    """Adds a `bias_pred` column (the model's full prediction: mean + movie term
    + user term) and a `residual` column if a rating column exists. Works for
    both training data (to produce ALS's residual target) and probe data (to
    score the bias-only rungs)."""
    out = add_time_bin(df, date_col=date_col, bin_days=bias_model["bin_days"])
    out = out.join(bias_model["movie_bias_overall"], "movie_id", "left")
    if bias_model["temporal"] and bias_model["movie_bias_binned"] is not None:
        out = out.join(bias_model["movie_bias_binned"], ["movie_id", "time_bin"], "left")
        out = out.withColumn(
            "movie_term",
            F.coalesce(F.col("movie_bias"), F.col("movie_bias_overall"), F.lit(0.0)),
        )
    else:
        out = out.withColumn(
            "movie_term", F.coalesce(F.col("movie_bias_overall"), F.lit(0.0))
        )

    out = out.join(bias_model["user_model"], "user_id", "left")
    out = out.withColumn(
        "user_term",
        F.coalesce(
            F.col("user_a") + F.col("user_b") * (F.col("day_num") - F.col("user_tbar")),
            F.lit(0.0),
        ),
    )
    out = out.withColumn(
        "bias_pred",
        F.lit(bias_model["global_mean"]) + F.col("movie_term") + F.col("user_term"),
    )
    if "rating" in out.columns:
        out = out.withColumn("residual", F.col("rating") - F.col("bias_pred"))
    return out


def clip_to_scale(df, col_name, out_name=None, low=1.0, high=5.0):
    """Ratings live on [1, 5]; every final prediction gets clipped there."""
    out_name = out_name or col_name
    return df.withColumn(
        out_name, F.least(F.greatest(F.col(col_name), F.lit(low)), F.lit(high))
    )
