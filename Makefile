# Makefile for red-agent

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 .
	poetry run mypy red_agent/

check: format lint

run:
	poetry run python red_agent/arena/run_arena.py

evaluate:
	poetry run python -c "from red_agent.utils.aggregate import evaluate_all_rounds; evaluate_all_rounds()"

clean:
	rm -rf logs/round_*
	rm -f logs/transcript.txt
	rm -f logs/complete_evaluation.csv
	rm -f logs/*.txt
