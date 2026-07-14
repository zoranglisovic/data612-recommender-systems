"""
Parses the full, unfiltered Netflix Prize corpus (all four combined_data files)
into a pandas DataFrame, then converts to Spark.

Same movie-header-tracking logic as Project 1 / Project 5's parse_netflix_file
(and the planning doc's own full-corpus EDA scan) -- extended to all four files,
with the date field kept (needed for temporal bias) and no popularity/activity
filtering, since the full corpus is this project's "additional data" element.

Runs as a single-threaded Python scan, not a Spark job -- the movie-header
line has to be tracked sequentially across each file, and a naive Spark window
over all ~100M rows to forward-fill it is a known anti-pattern (unpartitioned
global sort). The planning doc's own EDA cell already proved a full sequential
scan of all four files completes in tolerable time.
"""

import os
import time

import numpy as np
import pandas as pd

COMBINED_FILES = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]


def _flush_chunk(movie_ids, user_ids, ratings, dates):
    return pd.DataFrame(
        {
            "movie_id": np.array(movie_ids, dtype=np.int32),
            "user_id": np.array(user_ids, dtype=np.int32),
            "rating": np.array(ratings, dtype=np.int8),
            "date": np.array(dates, dtype="datetime64[D]"),
        }
    )


def parse_netflix_chunks(data_path, files=None, chunk_rows=5_000_000, progress=True):
    """Generator yielding pandas DataFrames of at most chunk_rows ratings each.

    The v1 parse-everything-into-Python-lists approach crashed the Databricks
    driver (OOM) on the full 100M-row corpus -- ~100M small Python objects per
    column costs 15-20GB+ of Python heap, on a driver whose 32GB is shared with
    the Spark JVM. Chunked parsing bounds peak memory at one chunk (~5M rows,
    a few hundred MB) regardless of corpus size. The staged 24M-row run survived
    v1 only because a quarter of the corpus still fit; this works at any scale.
    """
    files = files or COMBINED_FILES
    for fname in files:
        filepath = os.path.join(data_path, fname)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Could not find {filepath}. Point data_path at a local copy of the "
                "Netflix Prize data (not included in this repo)."
            )
        t0 = time.time()
        current_movie = None
        file_rows = 0
        movie_ids, user_ids, ratings, dates = [], [], [], []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    current_movie = int(line[:-1])
                else:
                    user_id, rating, date = line.split(",")
                    movie_ids.append(current_movie)
                    user_ids.append(int(user_id))
                    ratings.append(int(rating))
                    dates.append(date)
                    file_rows += 1
                    if len(movie_ids) >= chunk_rows:
                        yield _flush_chunk(movie_ids, user_ids, ratings, dates)
                        movie_ids, user_ids, ratings, dates = [], [], [], []
        if movie_ids:
            yield _flush_chunk(movie_ids, user_ids, ratings, dates)
        if progress:
            print(f"{fname}: {file_rows:,} ratings in {time.time() - t0:.1f}s")


def convert_to_parquet(data_path, parquet_path, files=None, chunk_rows=5_000_000,
                       progress=True):
    """One-time conversion: raw Netflix text format -> chunked Parquet.

    Run once; every pipeline run afterward (including every sweep fit) does
    spark.read.parquet(parquet_path) instead of re-parsing ~2GB of text --
    faster, and immune to the driver-OOM failure mode above. Parquet files are
    written per-chunk so peak memory stays bounded.
    """
    os.makedirs(parquet_path, exist_ok=True)
    total = 0
    for i, chunk in enumerate(parse_netflix_chunks(data_path, files=files,
                                                   chunk_rows=chunk_rows,
                                                   progress=progress)):
        out_file = os.path.join(parquet_path, f"part_{i:04d}.parquet")
        chunk.to_parquet(out_file, index=False)
        total += len(chunk)
        if progress:
            print(f"  wrote {out_file} ({len(chunk):,} rows, {total:,} total)")
    return total


def parse_netflix_full(data_path, files=None, progress=True):
    """Full corpus as one pandas DataFrame. Fine for subsets (tests, staged
    runs); do NOT call on all four files on a Databricks driver -- use
    convert_to_parquet + spark.read.parquet instead (see docstrings above)."""
    chunks = list(parse_netflix_chunks(data_path, files=files, progress=progress))
    return pd.concat(chunks, ignore_index=True) if chunks else _flush_chunk([], [], [], [])


def to_spark_df(spark, pandas_df):
    """Convert the parsed pandas DataFrame to a Spark DataFrame.

    Arrow must be enabled on the SparkSession (spark.sql.execution.arrow.pyspark.enabled)
    for this conversion to be fast at ~100M rows -- without it, this falls back to a
    much slower row-by-row conversion.
    """
    return spark.createDataFrame(pandas_df)


def load_movie_titles(data_path, filename="movie_titles.csv"):
    """movie_titles.csv is (movie_id, year, title) but titles can contain commas,
    so this can't be a plain pd.read_csv -- split on the first two commas only."""
    filepath = os.path.join(data_path, filename)
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, year, title = line.strip().split(",", 2)
            rows.append((int(movie_id), year, title))
    return pd.DataFrame(rows, columns=["movie_id", "year", "title"])


def _parse_eval_file(filepath, has_date):
    """Shared logic for probe.txt (movie_id, user_id) and qualifying.txt
    (movie_id, user_id, date) -- both use the same movie-header block format
    as combined_data, just without a rating column."""
    movie_ids, user_ids, dates = [], [], []
    current_movie = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                current_movie = int(line[:-1])
            else:
                if has_date:
                    user_id, date = line.split(",")
                    dates.append(date)
                else:
                    user_id = line
                movie_ids.append(current_movie)
                user_ids.append(int(user_id))

    data = {
        "movie_id": np.array(movie_ids, dtype=np.int32),
        "user_id": np.array(user_ids, dtype=np.int32),
    }
    if has_date:
        data["date"] = pd.to_datetime(dates)
    return pd.DataFrame(data)


def load_probe_pairs(data_path, filename="probe.txt"):
    """(movie_id, user_id) pairs whose true rating already exists in combined_data --
    the held-out evaluation set. No rating/date column in the source file."""
    return _parse_eval_file(os.path.join(data_path, filename), has_date=False)


def load_qualifying_pairs(data_path, filename="qualifying.txt"):
    """(movie_id, user_id, date) triples the original competition scored submissions
    against. Netflix never published the true ratings, so this can only be used to
    generate a prediction file in the right format, not to compute RMSE."""
    return _parse_eval_file(os.path.join(data_path, filename), has_date=True)
