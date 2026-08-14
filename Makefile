.PHONY: data train evaluate test app clean help

help:
	@echo "SwitchRank Makefile commands:"
	@echo "  make data      - Download and prepare WDC and AccessGUDID datasets"
	@echo "  make train     - Run baseline, linkage, and supervised model training"
	@echo "  make evaluate  - Run hard negatives, calibration, and transfer evaluations"
	@echo "  make test      - Run complete pytest suite"
	@echo "  make app       - Start FastAPI application server"
	@echo "  make clean     - Remove cached files and temporary artifacts"

data:
	uv run python scripts/download_wdc.py
	uv run python scripts/download_gudid.py
	uv run python scripts/prepare_data.py

train:
	uv run python experiments/run_baselines.py
	uv run python experiments/run_linkage.py
	uv run python experiments/run_supervised.py

evaluate:
	uv run python experiments/run_hard_negatives.py
	uv run python experiments/run_calibration.py
	uv run python experiments/run_transfer.py

test:
	uv run pytest tests/ -v

app:
	uv run uvicorn switchrank.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
