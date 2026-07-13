# DATA 612 Final Project — Databricks Pipeline

A "try it and see" pipeline for the full-scale temporal-bias-adjusted ALS
recommender, set up the same way Project 6's `aws_deployment/` folder was:
modular scripts + a runnable orchestration file + a README, so the actual
system can be run and timed, not just planned. The difference is what it's
being run against -- Project 6 deployed an already-trained model to AWS for
serving; this trains the Final Project's model on Databricks and reports how
long each stage takes on the real ~100M-row corpus.

## Why this exists

The planning doc's timing estimates were extrapolated from a synthetic
smoke test (200K and 2M synthetic rows on this same cluster type — see the
session notes). The only way to know if the full corpus actually finishes,
and how long it takes, is to run it. This folder is that attempt.

## Files

- `parser.py` — parses the full, unfiltered Netflix Prize corpus (all four
  `combined_data_*.txt` files) into a pandas DataFrame, then converts to Spark.
  Same movie-header-tracking logic already proven in Project 1/5 and the
  planning doc's own full-corpus EDA scan, extended to all four files with the
  date field kept. Also parses `probe.txt`, `qualifying.txt`, and `movie_titles.csv`.
- `temporal_bias.py` — global mean + time-binned movie bias + user bias, the
  exact mechanics already smoke-tested successfully on this cluster (including
  the RDD-level operation the professor flagged as a possible serverless risk
  — it did not error at up to 2M synthetic rows). v1 only: per-user bias here
  is a plain per-user mean, not yet the continuous smooth-trend regression the
  planning doc describes.
- `train_als.py` — fits ALS on the bias-adjusted residual (defaults are
  Project 5's winning config: rank=20, regParam=0.1 — a starting point the
  plan calls for re-sweeping on this corpus's own validation slice, not a
  final answer) and saves the model + bias tables so evaluation doesn't need
  to refit.
- `evaluate.py` — splits `probe.txt` pairs out of training, scores RMSE
  against their true ratings (already present in `combined_data`), and reports
  the `coldStartStrategy='drop'` rate honestly rather than hiding it.
- `00_full_pipeline_databricks.py` — the orchestration notebook. Written with
  `# Databricks notebook source` / `# COMMAND ----------` markers so Databricks
  imports it directly as a multi-cell notebook (via Repos, or Workspace ->
  Import -> file). Runs stages 1-6 end to end with wall-clock timing and row
  counts printed at every stage, so a slow or failing stage is visible
  immediately instead of hiding behind Spark's lazy evaluation.

## What this version does NOT yet do

Scoped deliberately small to answer "does it run, how long does it take" first:
- No `surprise` SVD baseline or blend regression yet (planning doc's Step 3) —
  add once the ALS-only timing is known.
- No precision@k / recall@k / coverage yet — RMSE only, for now.
- No `qualifying.txt` prediction file generation yet.
- Per-user bias is a plain mean, not the planning doc's continuous per-user
  time-drift regression.

## Setup — getting the data onto Databricks (done via CLI, 2026-07-13)

The Databricks CLI is authenticated against the workspace via Entra ID OAuth
(PATs are admin-disabled in the CUNY tenant; `databricks auth login` with the
existing browser session works instead). Profile name: `data612`.

```bash
export DATABRICKS_CONFIG_PROFILE=data612
databricks fs ls dbfs:/netflix_prize_data          # verify the upload
```

All seven files (`combined_data_1-4.txt`, `probe.txt`, `qualifying.txt`,
`movie_titles.csv`, ~2GB total) were uploaded from
`Resources/Netflix Prize data/` with `databricks fs cp`. `DATA_PATH` in the
orchestration notebook already points at `/dbfs/netflix_prize_data`.

Notes on options that did NOT work in this workspace, for the record:
- **Unity Catalog Volumes** — no metastore assigned to this workspace.
- **PAT tokens** — disabled at the org level; Entra ID OAuth is the way in.
- The DATA607-era `/mnt/myblob/` blob mount is still live and writable, but
  plain DBFS was chosen to avoid depending on a mount from an old course.

## Running it

1. Start the `data612 final compute` cluster (32GB/8 cores, `Standard_D8ds_v4`
   — the largest node type this Azure for Students subscription's vCPU quota
   actually allows; see session notes for why the v5 series and the 16-core
   v4 option are both blocked).
2. Import this folder into Databricks (Repos, connected to the
   `data612-recommender-systems` GitHub repo once pushed, or Workspace ->
   Import for a one-off run).
3. Open `00_full_pipeline_databricks.py`, update `DATA_PATH` to wherever the
   data actually landed (see above), attach the cluster, **Run All**.
4. Watch the printed timings per stage. If a stage hangs or errors, that's a
   real, useful data point — document it honestly rather than treating it as
   a bug to hide (the professor's own standing guidance: "if you have any
   issues with any of your submission, just make sure you document it well").
5. **Terminate the cluster when done** — it does not auto-stop quickly enough
   to avoid burning Azure for Students credit unnecessarily on an idle full-size
   node (30 min idle timeout is the default, but don't rely on it for a
   multi-hour credit budget).

## What to expect, going in

From the synthetic smoke test on this same 8-core/32GB cluster:

| Rows | Temporal bias | RDD op | ALS fit | Total |
|---|---|---|---|---|
| 200K (4-core/16GB) | 1.76s | 6.25s | 24.8s | 64.4s |
| 2M (8-core/32GB) | 4.66s | 14.67s | 38.8s | 106.7s |

10x more data cost only ~1.7x more wall-clock time in that test — a good sign,
but not a promise. The real corpus is ~50x larger again than the 2M-row test,
uses real (heavily skewed) data instead of uniform-random synthetic data, and
Spark's per-job fixed overhead matters less as data size grows, so this
extrapolation could be optimistic. Expect the real run to land somewhere
between "surprisingly fine" and "needs another round of staging at a
25M-row single-file subset before attempting the full 100M rows." Both
outcomes are useful information — that's the point of trying it.
