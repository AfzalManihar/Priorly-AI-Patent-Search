.PHONY: setup setup-ml sample serve test check eval real-data docker clean

setup:            ## install backend dependencies (offline backends)
	pip install -r backend/requirements-dev.txt

setup-ml:         ## add sentence-transformers for the bge backends
	pip install -r backend/requirements-ml.txt

sample:           ## regenerate the bundled sample corpus and queries
	cd backend && python -m ingest.make_sample_data

serve:            ## API and UI on http://localhost:8000
	cd backend && uvicorn app.main:app --reload

test:
	cd backend && python -m pytest -q

check:            ## everything CI runs: backend tests and a syntax check of the front-end scripts
	cd backend && python -m pytest -q
	for f in frontend/js/*.js; do node --check "$$f"; done

docker:           ## API plus Qdrant in containers on http://localhost:8000
	docker compose up --build

clean:            ## remove caches and the local database
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache backend/data/priorly.db

eval:             ## benchmark on the configured corpus (writes benchmark.json)
	cd backend && python -m eval.run_eval

real-data:        ## see README, "Using real data"
	@echo "1) python -m ingest.download --help"
	@echo "2) python -m ingest.build_corpus --help"
	@echo "3) python -m eval.build_queries --help"
