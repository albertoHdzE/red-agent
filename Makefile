# Makefile for red-agent

.PHONY: format lint test check run evaluate clean precommit fix-hooks commit-force

format:
	@echo "🧹 Formatting code with black..."
	poetry run black red_agent/
	@echo "🧹 Formatting code with isort..."
	poetry run isort red_agent/
	@echo "🧹 Removing trailing whitespace..."
	find . -name "*.yaml" -o -name "*.jinja" | xargs -I{} sed -i '' 's/[[:space:]]*$$//' {}
	@echo "🧹 Ensuring files end with newline..."
	find . -name "*.yaml" -o -name "*.jinja" | xargs -I{} sh -c '[ -z "$$(tail -c 1 "{}")" ] || echo "" >> "{}"'
	@echo "✅ Formatting complete!"

lint:
	poetry run flake8 red_agent/
	poetry run mypy red_agent/

test:
	poetry run pytest

check: format lint

# Add a precommit command to run all pre-commit checks
precommit:
	@echo "🔍 Running pre-commit checks..."
	poetry run pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

run:
ifeq ($(mode),test)
	@echo "🧪 Running in test mode with random topic..."
	poetry run python red_agent/arena/run_arena.py --test
else
	@echo "🚀 Running in full mode with all topics..."
	poetry run python red_agent/arena/run_arena.py
endif

evaluate:
	poetry run python -c "from red_agent.utils.aggregate import evaluate_all_rounds; evaluate_all_rounds()"

clean:
	rm -f logs/transcript.txt
	rm -f logs/evaluation.csv
	rm -f logs/debug.log

# Fix the specific flake8 issue in langgraph_arena.py
fix-hooks:
	@echo "🔧 Fixing pre-commit hook issues..."
	@echo "🔧 Fixing B023 error in langgraph_arena.py..."
	@sed -i '' 's/global_referee = RefereeAgent()/global_referee_instance = RefereeAgent()/' red_agent/arena/langgraph_arena.py
	@sed -i '' 's/global_referee/global_referee_instance/g' red_agent/arena/langgraph_arena.py
	@echo "✅ Fixed hook issues!"

# Add a commit command that stages all changes, formats, and commits
commit:
	@echo "🔄 Staging all changes..."
	git add .
	@echo "🧹 Running format checks..."
	$(MAKE) format
	@echo "🔄 Staging formatted files..."
	git add .
	@echo "📝 Running pre-commit checks..."
	poetry run pre-commit run --all-files
	@echo "🔄 Staging any additional changes..."
	git add .
	@echo "🌿 Committing on branch: $$(git rev-parse --abbrev-ref HEAD)"
	@echo "📝 Committing changes..."
	git commit -m "$(message)"

commit-force:
	@echo "🔄 Staging all changes..."
	git add .
	@echo "🧹 Running format checks..."
	$(MAKE) format
	@echo "🔄 Staging formatted files..."
	git add .
	@echo "🌿 Committing on branch: $$(git rev-parse --abbrev-ref HEAD)"
	@echo "📝 Committing changes (bypassing hooks)..."
	git commit -m "$(message)" --no-verify
