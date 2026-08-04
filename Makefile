.PHONY: setup test run clean lint

setup:
	python -m venv .venv
	.\.venv\Scripts\pip install -r requirements.txt
	.\.venv\Scripts\pip install -e .

test:
	pytest tests/ --cov=driftx --cov-report=term-missing

run:
	python scripts/run_experiment.py --config configs/default.yaml

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
