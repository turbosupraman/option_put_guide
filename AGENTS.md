# Repository Guidelines

## Project Structure & Module Organization
This repository is currently minimal (only `.gitignore` and a local `.venv`). As code is added, use a simple, predictable layout:
- `src/` for application/library code.
- `tests/` for automated tests.
- `scripts/` for one-off utilities (data fetch, setup).
- `docs/` for documentation and design notes.
Keep imports explicit and avoid circular dependencies.

## Build, Test, and Development Commands
- `python3 -m venv .venv` creates the local virtual environment.
- `source .venv/bin/activate` activates it for the current shell.
- `python -m pip install -r requirements.txt` installs dependencies (when a `requirements.txt` exists).
- `python -m pytest` runs tests (once tests are present).
There is no build step yet; add one here if you introduce packaging or CI.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 style, and type hints where practical.
- Names: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Filenames: lowercase with underscores (e.g., `data_loader.py`).
If a formatter/linter is added (e.g., `black`, `ruff`), follow its defaults.

## Testing Guidelines
- Preferred framework: `pytest`.
- Test files: `tests/test_*.py`; test functions: `test_*`.
- Keep tests deterministic; avoid network calls unless explicitly mocked.
No coverage target is set yet; document one if the suite grows.

## Commit & Pull Request Guidelines
- Commit messages are short, lowercase, and imperative (e.g., `add ingest script`), matching the existing history.
- Keep commits focused on a single change.
- PRs should include a brief summary, testing notes, and linked issues if applicable.
Include screenshots only if visual output is added later.

## Environment & Configuration
- Do not commit `.venv/` or secrets.
- If you add environment variables, document them and use a local `.env` (gitignored).
