.PHONY: all test fast clean
all:
	python run_all.py
fast:
	python run_all.py --skip-network
test:
	pytest -q
clean:
	rm -rf data/processed/* outputs/results/* outputs/figures/* outputs/tables/*
