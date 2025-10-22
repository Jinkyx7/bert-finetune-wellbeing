# Training Program Overview
This repository houses the training pipeline for fine-tuning BERT on wellbeing classification tasks. The guidance below outlines how to structure modules, manage experiments, and contribute changes so that model training remains reproducible and consistent.

# Repository Guidelines

## Project Structure & Module Organization
The project targets BERT fine-tuning for wellbeing classification. Keep all Python packages inside `src/`, splitting code into `src/data`, `src/models`, `src/training`, and `src/utils` as the pipeline grows. Mirror that layout in `tests/` (for example, `tests/training/test_scheduler.py`) to keep unit coverage aligned with runtime modules. Store exploratory notebooks under `notebooks/` and avoid committing executed output cells. The shared dataset lives in `dataset/combined_labels.csv`; if you create derived subsets, write them to `dataset/processed/` with clear versioning notes in the PR.

## Build, Test, and Development Commands
Work inside an isolated environment: `python -m venv .venv && source .venv/bin/activate`. Install dependencies with `pip install -r requirements.txt` and update that file whenever tool versions change. Run static checks using `ruff check src tests` and formatters with `black src tests`. Execute training locally through `python -m src.training.run --config configs/base.yaml`, which should read data from `dataset/` and persist checkpoints to `artifacts/`. Use `jupyter notebook notebooks/exploration.ipynb` for ad-hoc validation runs.

## Coding Style & Naming Conventions
Write Python 3.10+ code with four-space indentation, explicit type hints, and PEP 8 spacing. Modules, functions, and variables stay in `snake_case`; classes use `PascalCase`; constants remain uppercase. Keep pure functions side-effect free and document their behavior with concise docstrings. Name configuration files with kebab-case YAML (e.g., `configs/base.yaml`) and prefer dependency injection over global state for trainer components.

## Testing Guidelines
Use `pytest` and name tests `test_<unit>_<behavior>` so failures read cleanly. Common fixtures belong in `tests/conftest.py` and should provide small slices of `combined_labels.csv` for deterministic assertions. Target at least 85% line coverage via `pytest --cov=src` before submitting a PR. Add integration smoke tests under `tests/integration/` whenever you introduce new training loops, schedulers, or data preprocessing steps.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat: add attention pooling head`) and keep subject lines within 50 characters. Include verification details in the commit body, especially notes about dataset or checkpoint changes. Before opening a PR, ensure `ruff`, `black`, and `pytest` all pass and paste command summaries in the description. PRs should link related issues, summarize evaluation metrics (macro-F1, accuracy), and mention any new artifacts or configuration files. Request at least one review and rebase on `main` prior to merging.
