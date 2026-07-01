## Summary

Ports the entire codebase from two monolithic scripts into a standard Python package structure.

### Changes

- **Split** 2162-line `reforge.py` into 12 focused modules under `src/reforge/` (config, profiles, display, dataset, tokenization, training, early_stopping, reporting, utils, dashboard, __main__)
- **PII removed** — all hardcoded `D:\...` drive paths replaced with cross-platform defaults (`REFORGE_MODELS_DIR`, `REFORGE_OUTPUT_DIR` env vars, `~/HF_Models`, `~/Reforge_Output`)
- **Added** `pyproject.toml` for installable package (`pip install -e .`)
- **Updated README** with new project structure, usage (`python -m reforge`), and environment variables
- **Updated** GitHub repo description and topics

### How to run

```
# CLI
python -m reforge --help

# Dashboard
python -m reforge.dashboard

# Or install globally
pip install -e .
reforge --help
```
