.PHONY: install test run clean lint

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

run:
	streamlit run app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

lint:
	flake8 src/ --max-line-length=120 --ignore=E501

docker-build:
	docker build -t project_11_311_service_request_router .

docker-run:
	docker run -p 8501:8501 project_11_311_service_request_router
