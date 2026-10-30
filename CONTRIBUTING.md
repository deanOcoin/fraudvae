# Contributing

The short version of how the project is worked on, so a change fits in.

## Setting up

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # the cpu wheel; plain pip pulls the cuda build
pip install -e ".[dev]"
make fix-omp                                                          # macOS only, once per venv
```

## Tests

```bash
make test        # the whole suite: synthetic data, a two-epoch model, the api, an in-process ssh server; about a minute
make test-fast   # pytest -m "not slow": everything that does not train a model first, in seconds
make lint        # ruff, with the rules in pyproject.toml; fails on a finding
```

No real data is needed: `tests/conftest.py` builds a throwaway data / runs / home tree and
removes it at exit. A test that needs the trained smoke model (the `trained_run` or `client`
fixture) is marked `slow` automatically, so the quick loop is the one to run while editing
something that does not touch training. A new feature gets a test in the module it belongs to.

## Conventions

- Python 3.10 is supported and CI runs it: no nested-quote f-strings, nothing 3.12-only.
- The handlers that score (`/api/score`, `/api/score/batch`) are plain `def`, never `async def`:
  uvicorn runs one worker, and a sync handler goes to the thread pool.
- Every byte the site serves comes from its own origin: no CDN, no external font or script.
  Adding one is a deliberate change to `SECURITY.md`.
- The UI has to fit every width from 320 px without horizontal scroll, work on touch, and be
  reachable from the keyboard; retake `docs/img/*.png` after a visible change.
- Documentation changes with the code, in the same change: the README, `SECURITY.md`,
  `docs/DEPLOY.md`, `docs/TRAINING_NOTES.md`, the story pages, whichever describes what moved.
  Figures in them are read from `eval.json` files or measured, never typed from memory.
- One change per commit, with a prose message: a lowercase subject that says what and why, a
  blank line, then paragraphs. A round of work becomes several commits.
