# Entry point. Use this rather than invoking the scripts directly — execute
# permissions do not survive a zip, a file copy or some checkouts, so
# `./run_smoke_test.sh` fails on a fresh copy with "Permission denied".

.PHONY: help setup verify test guard stats-check quant-check rehearse-cpu plan status rehearse sweep clean-results paper check-mock

help:
	@echo "make setup          install dependencies and restore execute bits"
	@echo "make verify         is this copy complete? (run first on any new copy)"
	@echo "make test           full smoke test (~20s, no GPU/API/device needed)"
	@echo "make quant-check    exact llama.cpp quantizer tests (needs torch)"
	@echo "make rehearse-cpu   REAL training + evaluation code on a tiny model (needs torch)"
	@echo ""
	@echo "make plan           show the experiment sweep, run nothing"
	@echo "make rehearse       run the whole sweep on mock data (~10s)"
	@echo "make sweep          RUN THE REAL SWEEP (resumable)"
	@echo "make status         what is done, what is pending"
	@echo ""
	@echo "make guard          safety guard tests only (no dependencies at all)"
	@echo "make stats-check    validate Dunn's test and the kappas"
	@echo "make paper          build the conference paper"
	@echo "make check-mock     fail if any mock figure is present in results/"
	@echo "make clean-results  delete generated results and figures"

setup:
	@chmod +x run_smoke_test.sh mobile/sync_rules.sh 2>/dev/null || true
	pip install -r requirements.txt
	@echo "\nReady. Now run: make test"

verify:
	@python3 scripts/verify_repo.py

quant-check:
	@python3 src/test_quant_sim.py

rehearse-cpu:
	@python3 scripts/cpu_rehearsal.py

plan:
	@python3 src/run_pipeline.py --dry-run

status:
	@python3 src/run_pipeline.py --status

rehearse:
	@python3 src/run_pipeline.py --mock

sweep:
	@python3 src/run_pipeline.py

test:
	@chmod +x run_smoke_test.sh 2>/dev/null || true
	@bash run_smoke_test.sh

guard:
	@python3 src/safety/test_guard.py

stats-check:
	@python3 src/test_stats.py

paper: check-mock
	@$(MAKE) -C paper

# Mock figures must never reach the paper. They are watermarked, but a
# watermark is a last line of defence, not a first one.
check-mock:
	@python3 scripts/check_mock.py results/figures

clean-results:
	rm -rf results/figures results/*.jsonl results/*.json runs
	@echo "results cleared"
