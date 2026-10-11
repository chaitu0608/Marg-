PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

.PHONY: test bench world train serve loadtest pipeline docs clean-generated verify lint

test:
	$(PYTEST) -q

world:
	$(PYTHON) -m bench.plots --world --seed 7 --horizon 20000

train:
	$(PYTHON) -m marg.fraud.train --data data/creditcard.csv --seed 7

loadtest:
	$(PYTHON) -m bench.run_loadtest

pipeline:
	$(PYTHON) -m marg.pipeline

bench:
	$(PYTHON) -m bench.run_routing --seed 7 --seeds 20 --horizon 20000
	$(PYTHON) -m bench.run_window_sweep
	$(PYTHON) -m marg.fraud.train --data data/creditcard.csv --seed 7
	$(PYTHON) -m bench.run_loadtest
	$(PYTHON) -m marg.pipeline
	$(PYTHON) scripts/generate_readme.py

serve:
	$(PYTHON) -m uvicorn marg.fraud.serve:app --host 127.0.0.1 --port 8000

docs:
	$(PYTHON) scripts/generate_readme.py

clean-generated:
	$(PYTHON) scripts/clean_generated.py

verify: clean-generated world test bench
	$(PYTHON) scripts/generate_readme.py --check
	$(PYTHON) -m ruff check src bench tests scripts
