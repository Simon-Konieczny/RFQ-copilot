# Run inside the conda env: `conda activate rfqco` first.
.PHONY: help install test lint check demo demo-v2 eval eval-quick report clean clear-cache

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:  ## editable install with dev tools
	pip install -e ".[dev]"

test:  ## unit tests (offline, scripted LLM)
	pytest -q

lint:  ## ruff, lint only (never reformats)
	ruff check .

check: lint test  ## lint + tests

demo:  ## chat with the agent as a client (uses the LLM, cached)
	python -m rfq_copilot

demo-v2:  ## same, with the v2 LangGraph agent
	python -m rfq_copilot --v2

eval:  ## v1 vs v2 on the eval set -> eval/results.md + eval/report.html (uses the LLM, cached)
	python eval/run_eval.py

eval-quick:  ## same on the 27-row dev split only, no dialogues
	python eval/run_eval.py --split dev --no-dialogues

report:  ## open the last eval's report in the browser
	python -m webbrowser -t "file://$(CURDIR)/eval/report.html"

clean:  ## remove build/test caches (keeps the LLM cache)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache src/*.egg-info

clear-cache:  ## delete cached LLM responses: next runs hit the API (free-tier quota!)
	rm -rf .cache/llm
