.PHONY: setup test run clean lint \
        run-curves run-sensitivity run-null run-changepoints run-ablation \
        run-multi run-all-honest

setup:
	python -m venv .venv
	.\.venv\Scripts\pip install -r requirements.txt
	.\.venv\Scripts\pip install -e .

test:
	pytest tests/ --cov=driftx --cov-report=term-missing

run:
	python scripts/run_experiment.py --config configs/default.yaml

# --- Honest evaluation targets (run in order) ---

# Tune on a held-out stream FIRST, then run frozen config
tune:
	python scripts/tune_on_holdout.py --source synthetic --seeds 2

# Main benchmark (use frozen.yaml after tuning; falls back to default.yaml)
run-main:
	python scripts/run_experiment.py --config configs/frozen.yaml 2>NUL || \
	python scripts/run_experiment.py --config configs/default.yaml

# Cross-dataset (fraud + intrusion: 3 seeds; electricity separate)
run-multi:
	python scripts/run_multi_dataset.py --datasets Fraud Intrusion --seeds 3
	python scripts/run_multi_dataset.py --datasets Electricity --seeds 3

# Accuracy-vs-retrain Pareto curves for each policy (Issue 3 fix)
run-curves:
	python scripts/run_policy_curves.py --seeds 3

# Sensitivity grid on held-out stream — NOT on eval streams (Issue 3 fix)
run-sensitivity:
	python scripts/run_sensitivity.py

# Null-stream false-positive control (strengthened concept)
run-null:
	python scripts/run_null_stream.py --n-windows 20 --seeds 5

# Known changepoint detection (standard + concept-only variant)
run-changepoints:
	python scripts/run_known_changepoints.py --n-post-cp 5 --seeds 3
	python scripts/run_known_changepoints.py --n-post-cp 5 --seeds 3 --concept-only

# Ablation study
run-ablation:
	python scripts/run_ablation.py

# Full honest re-run sequence (all experiments)
run-all-honest: run-main run-multi run-curves run-sensitivity run-null run-changepoints run-ablation

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
