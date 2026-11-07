.PHONY: env install data data-smoke starter-kit build run smoke test lint paper clean-runs

PY ?= python
DOMAINS ?= all
SPLIT ?= dev
SUB ?= 1a
EXP ?= configs/experiments/e0_bm25.yaml

env:            ## create the conda environment
	mamba env create -f environment.yml || conda env create -f environment.yml

install:        ## editable install of the package
	$(PY) -m pip install -e .

data:           ## download all Track 1 domains
	$(PY) scripts/download_data.py

data-smoke:     ## download two small domains for a quick test
	$(PY) scripts/download_data.py --domains iota hsm

starter-kit:    ## fetch the official starter kit (scorer, format checker)
	git clone --depth 1 https://github.com/DataScienceUIBK/RETECO.git third_party/RETECO

build:          ## build indexes / metadata / KG: make build EXP=... DOMAINS="iota hsm"
	$(PY) -m chronofuse.cli build --config $(EXP) --domains $(DOMAINS)

run:            ## retrieve + score: make run EXP=... SPLIT=dev SUB=1a
	$(PY) -m chronofuse.cli run --config $(EXP) --split $(SPLIT) --subtrack $(SUB) --domains $(DOMAINS)

smoke:          ## end-to-end on iota with BM25
	$(MAKE) build DOMAINS=iota && $(MAKE) run DOMAINS=iota SPLIT=train

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts

paper:          ## compile the ACL-format paper (needs a TeX distribution)
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean-runs:
	rm -rf results/runs/* results/metrics/*
