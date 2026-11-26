PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn
STREAMLIT ?= .venv/bin/streamlit

.PHONY: install lint typecheck train run-pipeline visuals test api ui dashboard verify clean

install:
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src ml pipelines scripts app tests
	$(RUFF) format --check src ml pipelines scripts app tests

typecheck:
	$(MYPY) src ml

train:
	$(PYTHON) ml/train.py

run-pipeline:
	$(PYTHON) scripts/run_pipeline.py --render-visuals

visuals:
	$(PYTHON) scripts/render_visuals.py

test:
	$(PYTEST) -v --cov=mall_seg

api:
	$(UVICORN) mall_seg.api:app --host 0.0.0.0 --port 8010

ui:
	$(STREAMLIT) run app/ui.py --server.port 8501 --server.address 0.0.0.0

dashboard:
	$(STREAMLIT) run app/dashboard.py --server.port 8501 --server.address 0.0.0.0

verify: lint typecheck run-pipeline test

clean:
	rm -rf artifacts/*.db artifacts/*.joblib artifacts/visuals/*.png artifacts/visuals/*.html .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
