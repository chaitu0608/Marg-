PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

.PHONY: test bench world train serve loadtest pipeline outage-detector outage-tradeoff fraud-sensitivity docs clean-generated verify lint ci mcp attest web3-test

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

outage-detector:
	$(PYTHON) -m bench.run_outage_detector --seed 7 --seeds 20 --horizon 20000

outage-tradeoff:
	$(PYTHON) -m bench.run_outage_tradeoff --seed 7 --seeds 20 --horizon 20000

fraud-sensitivity:
	$(PYTHON) -m bench.run_fraud_sensitivity --data data/creditcard.csv

mcp:
	$(PYTHON) -m marg.mcp_server

attest:
	$(PYTHON) scripts/create_attestation.py

web3-test:
	forge fmt --check
	forge test -vv

bench:
	$(PYTHON) -m bench.run_routing --seed 7 --seeds 20 --horizon 20000
	$(PYTHON) -m bench.run_outage_detector --seed 7 --seeds 20 --horizon 20000
	$(PYTHON) -m bench.run_outage_tradeoff --seed 7 --seeds 20 --horizon 20000
	$(PYTHON) -m bench.run_window_sweep
	$(PYTHON) -m marg.fraud.train --data data/creditcard.csv --seed 7
	$(PYTHON) -m bench.run_fraud_sensitivity --data data/creditcard.csv
	$(PYTHON) -m bench.run_loadtest
	$(PYTHON) -m marg.pipeline
	$(PYTHON) scripts/generate_readme.py

serve:
	$(PYTHON) -m marg.fraud.server

docs:
	$(PYTHON) scripts/generate_readme.py

clean-generated:
	$(PYTHON) scripts/clean_generated.py

verify: clean-generated world test bench
	$(PYTHON) scripts/generate_readme.py --check
	$(PYTHON) -m ruff check src bench tests scripts

lint:
	$(PYTHON) -m ruff check src bench tests scripts

ci: test lint
	$(PYTHON) scripts/generate_readme.py --check
