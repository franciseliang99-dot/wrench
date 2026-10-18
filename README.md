# Wrench

Wrench is a maintenance assistant for a train's air compressor. It tells the technician three things:

1. **Is this unit about to fail?** An alert level (ok / watch / act) for every 10 minutes of sensor data, with the sensors that drive it.
2. **What should I do about it?** Answers from the service manual and past maintenance records. Every claim carries a quote, and the quote is checked word for word against the source. A claim whose quote does not match is dropped.
3. **A work order to approve.** A draft built from the alert and the cited manual sections. A person approves it.

The alert model is a small convolutional autoencoder trained only on healthy data. **A language model never decides whether a unit is in alarm.** It only explains and searches.

## What runs where

| Part | Where |
|---|---|
| Autoencoder training and scoring of the full year | PyTorch on CPU (69.1 s for 6 epochs). The code also runs on an AMD Instinct GPU under ROCm, which PyTorch exposes as the `cuda` device; that run was not done |
| Answers and work-order steps | Gemma-4-31B-IT on AMD Token Factory, AMD's hosted OpenAI-compatible model API that runs on AMD GPUs. The key stays on the server (`WRENCH_TF_KEY`) |
| Web page and API | A small always-on host (the API needs only `fastapi`, `uvicorn` and `httpx`; about 84 MB resident) |

The shared Token Factory key allows 20 requests a minute and a daily spend cap, so the live model can be busy. When a call fails:

- Questions that were recorded from the same model are answered from the recording. The page marks them **RECORDED** with the time and the machine, and the recorded output goes through the same quote check as a live one.
- Other questions and all work orders return `generation_offline`. Work orders are never replayed.

## Results

All numbers come from `artifacts/`. They are written by the scripts below, not by hand.

**Data split by time**, fixed before any model was trained (`data/splits.json`): train Feb–Mar 2020 (no failures), validation Apr 1 – May 15 (failure 1), evaluation May 15 – Sep 2 (failures 2, 3, 4).

**Alerts** (`artifacts/evaluation.json`): first `act` alert compared with the start of each recorded failure.

| Failure | Segment | Start | First act alert | Lead |
|---|---|---|---|---|
| 1 | validation | 2020-04-18 00:00 | 2020-04-18 01:10 | during the failure |
| 2 | evaluation | 2020-05-29 23:30 | 2020-05-30 00:10 | during the failure |
| 3 | evaluation | 2020-06-05 10:00 | 2020-06-03 11:00 | 47.0 h early |
| 4 | evaluation | 2020-07-15 14:30 | 2020-07-14 03:20 | 35.2 h early |

False `act` alerts per week of normal operation: 0.71 on validation (3 episodes in 4.21 weeks), 0.36 on evaluation (4 episodes in 11.22 weeks).

**Retrieval** (`artifacts/retrieval.json`): on 20 hand-written questions, the right manual section is first 12/20 times and in the top 3 20/20 times. The refusal threshold τ = 3.758 (BM25 score) was set on 10 questions outside the manual's scope. It gets one of them wrong: "WD-200 warranty" matches the product name and is answered with an unrelated section.

## Limitations, stated plainly

- **Only four failures, all air leaks, on one compressor.** Two of the four were caught only after they started. Four events are not enough for a reliable average, so the table lists each one.
- **The evaluation segment was looked at once before tuning.** A first version was run on it. After that, all changes were based on the validation segment only, and the reported numbers come from a `--final` run. It is not a held-out set that was never seen.
- **The manual and the maintenance records are fictional**, written for this project (`corpus/README.md`). The retrieval questions were written by the same author as the manual, so the retrieval numbers are optimistic.
- **The alert model was trained and scored on CPU.** Only the language model runs on AMD GPUs. A live answer takes 15 to 35 seconds.
- **Priority 1 never fires on this data.** The rule (an `LPS` driver means "inspect before the next run") is covered by tests, but no `act` step in this data has `LPS` among its drivers.
- Work-order state lives in memory and resets when the server restarts.

## Run it

Python 3.11 or later.

```sh
pip install -e '.[pipeline,test]'
python scripts/fetch_data.py                    # download MetroPT-3 and check its sha256
python scripts/build_timeline.py --final        # train, set thresholds on validation, score, evaluate
python scripts/compact_timeline.py              # artifacts/timeline.min.json for the web page
python scripts/eval_retrieval.py                # retrieval hit rate and tau
pytest
```

Serve (the base install is enough):

```sh
pip install -e .
WRENCH_GENERATOR=stub uvicorn --factory wrench.api:app_from_env   # development: a CPU stub, not a model
```

For live answers set `WRENCH_TF_KEY` to an AMD Token Factory API key instead (`WRENCH_TF_MODEL` picks the model, default `Gemma-4-31B-IT`). A self-hosted vLLM server works too: `WRENCH_GPU_URL`, `WRENCH_GPU_KEY`, `WRENCH_GPU_CERT`, `WRENCH_GPU_MODEL` (see `wrench/generation.py`). `WRENCH_DEMO_PASSCODE` turns on approval, and approval is off when it is unset. `WRENCH_ORIGIN` sets the one allowed cross-site origin. Set `WRENCH_TRUST_PROXY=1` only behind a proxy that appends `X-Forwarded-For`.

## Data and license

Sensor data: **MetroPT-3**, UCI Machine Learning Repository, DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), licensed CC BY 4.0. Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., Gama, J. (2021). *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.* IEEE DSAA 2021. The data is downloaded by `scripts/fetch_data.py` and is not stored in this repository. The failure table in `data/failures.json` is copied from the dataset page.

Code, manual and records: MIT (`LICENSE`).
