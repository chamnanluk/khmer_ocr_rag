.PHONY: install test demo e1 e2 e3 e4 e5 clean
install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

demo:
	python scripts/demo_retrieval.py

e1:
	python -m khmer_ocr_rag.cli run configs/experiments/e1_rq1.yaml

e2:
	python -m khmer_ocr_rag.cli run configs/experiments/e2_rq2.yaml

e3:
	python -m khmer_ocr_rag.cli run configs/experiments/e3_rq3_rate_curve.yaml

e4:
	python -m khmer_ocr_rag.cli run configs/experiments/e4_rq3_matched_type.yaml

e5:
	python -m khmer_ocr_rag.cli run configs/experiments/e5_rq4.yaml

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
