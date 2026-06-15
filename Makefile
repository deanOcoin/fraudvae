# convenience targets, everything is also reachable as `fraudvae <cmd>` / python -m
PY ?= python3
PREPARED ?= default
CONFIG ?= cpu_small
CAP ?= 1500000

.PHONY: setup fix-omp data build prepare prepare-full train post serve test lint clean-runs

setup:
	$(PY) -m pip install -e ".[dev]"

# macOS: one OpenMP runtime for torch + lightgbm (see scripts/macos_openmp.sh)
fix-omp:
	PYTHON=$(PY) bash scripts/macos_openmp.sh

data:
	$(PY) -m fraudvae.data.fetch

build:
	$(PY) -m fraudvae.data.build

# capped train rows per source, what a laptop can chew through
prepare:
	$(PY) -m fraudvae.data.prepare --name $(PREPARED) --max-rows-per-source $(CAP)

prepare-full:
	$(PY) -m fraudvae.data.prepare --name full

train:
	$(PY) -m fraudvae.train --config $(CONFIG) --prepared $(PREPARED)

# booster + calibration + test eval again for an existing run: make post RUN=name
post:
	$(PY) -m fraudvae.cli post $(RUN) --config $(CONFIG) --prepared $(PREPARED)

serve:
	$(PY) -m fraudvae.cli serve

test:
	$(PY) -m pytest -q -p no:warnings

lint:
	$(PY) -m pyflakes fraudvae tests || true

clean-runs:
	rm -rf runs/smoke-* runs/test-run
