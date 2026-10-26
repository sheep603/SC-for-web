# Semantic Secrecy: Core Experiment Package

This directory is a clean, independent copy of the core experiment code and the MS MARCO data file used by the Web-search experiments. It does not contain analysis/report-generation scripts, figures, paper drafts, cached outputs, or the original working files.

## Layout

```text
src/
  web_search/
    secrecy_region.py
    web_search.py
    ablation.py
    query_channel.py
    prepare_region.py
    plot_region.py
  external_baselines/
    baseline_models.py
    run_baselines.py
  gridworld_legacy/
    channel.py, encoder.py, distortion.py, joint_region.py,
    encoding_search.py, semantic_set.py, set_search.py
data/msmarco/train.jsonl
results/                       # generated outputs
```

## Environment

Use Python 3.10 or newer. Install the dependencies from the package root:

```powershell
python -m pip install -r requirements.txt
```

## Web experiment order

Run the main finite-codebook experiment and application-level retrieval evaluation:

```powershell
python .\src\web_search\secrecy_region.py
python .\src\web_search\web_search.py
```

Run the existing three-scheme ablation and the separately scoped query-channel experiment when needed:

```powershell
python .\src\web_search\ablation.py
python .\src\web_search\query_channel.py
```

Run the Web-adapted external baseline experiment:

```powershell
python .\src\external_baselines\run_baselines.py
```

Generate the Web 3-D region data and figures after running the two main finite-codebook scripts:

```powershell
python .\src\web_search\prepare_region.py
python .\src\web_search\plot_region.py
```

All generated files, including plots, are written under `results/`; source data under `data/` is read-only by convention.

## Model and reproducibility notes

The external baseline mechanisms are implemented together in `src/external_baselines/baseline_models.py`, with orchestration in `run_baselines.py`. They are Web-interface adaptations, not full reproductions of the papers' private training code. No published model checkpoint is present in the source workspace. The PLM baseline falls back to a clearly identified TF-IDF/SVD proxy unless a local checkpoint is supplied via `WEB_PLM_MODEL_PATH`; proxy results must not be presented as results from the published PLM model.

To load a compatible local PLM checkpoint, install the optional packages with `python -m pip install -r requirements-plm.txt`, then set `WEB_PLM_MODEL_PATH` to the checkpoint directory before running the external baseline script. Installing these packages does not supply the checkpoint itself.

## Data

See `data/README.md` for the included MS MARCO file, provenance, size, and which split the current scripts use. The Web experiments sample eligible query records from this file; `dev.jsonl`, `dl20.jsonl`, and previously generated analysis tables are not required by the included core scripts.
