.PHONY: install ingest run test cov eval lint

install:
	pip install -r requirements-dev.txt

ingest:
	python -m src.rag.ingest --rebuild

run:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest

cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html

eval:
	python -m evaluation.run_eval

eval-fast:
	python -m evaluation.run_eval --retrieval-only

lint:
	ruff check src tests evaluation
