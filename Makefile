# Makefile for red-agent
#
# Targets:
#   - format: Formats code using black, isort, and fixes whitespace/newlines in YAML/Jinja files.
#   - lint: Runs flake8 and mypy for static code analysis.
#   - test: Runs unit tests using pytest (intended for unit testing, not for running debates).
#   - check: Runs format and lint targets.
#   - precommit: Runs pre-commit hooks on all files.
#   - run: Executes the debate system with various modes:
#       - Default: Runs debates for all topics with all agents.
#       - mode=test: Runs a single random topic with all agents.
#       - mode=wide: Runs specified number of topics with randomly selected agents.
#         Example: make run mode=wide topics=3 min-comments=3 num-agents=3
#       - mode=wide-2: Runs 2 topics with 2 randomly selected agents, each making at least 2 comments.
#         Example: make run mode=wide-2
#       - mode=wide-3: Runs 3 topics with 3 randomly selected agents, each making at least 3 comments.
#         Example: make run mode=wide-3
#   - evaluate: Aggregates and evaluates debate results.
#   - clean: Removes log files (transcripts, evaluation CSV, debug logs).
#   - fix-hooks: Fixes specific pre-commit hook issues (e.g., B023 in langgraph_arena.py).
#   - commit: Stages, formats, runs pre-commit checks, and commits changes.
#   - commit-force: Same as commit but bypasses pre-commit hooks.

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
	@if [ -n "$(mode)" ] || [ -n "$(topics)" ] || [ -n "$(min-comments)" ] || [ -n "$(num-agents)" ]; then \
		echo "❌ Error: The 'test' target is for running unit tests with pytest, not for running debates."; \
		echo "   Parameters like 'mode', 'topics', 'min-comments', or 'num-agents' are meant for the 'run' target."; \
		echo "   Use 'make run' instead, e.g., 'make run mode=wide topics=3 min-comments=3 num-agents=3'."; \
		exit 1; \
	fi
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
else ifeq ($(mode),wide)
	@echo "🌐 Running in wide mode with specified parameters..."
	poetry run python red_agent/arena/run_arena.py --wide \
		--topics=$(or $(topics),1) \
		--min-comments=$(or $(min-comments),3) \
		--num-agents=$(or $(num-agents),5)
else ifeq ($(mode),wide-2)
	@echo "🌐 Running in wide-2 mode (2 topics, 2 agents, min 2 comments each)..."
	poetry run python red_agent/arena/run_arena.py --wide \
		--topics=$(or $(topics),2) \
		--min-comments=$(or $(min-comments),2) \
		--num-agents=$(or $(num-agents),2)
else ifeq ($(mode),wide-3)
	@echo "🌐 Running in wide-3 mode (3 topics, 3 agents, min 3 comments each)..."
	poetry run python red_agent/arena/run_arena.py --wide \
		--topics=$(or $(topics),3) \
		--min-comments=$(or $(min-comments),3) \
		--num-agents=$(or $(num-agents),3)
else
	@echo "🚀 Running in full mode with all topics..."
	poetry run python red_agent/arena/run_arena.py
endif

evaluate:
	poetry run python -c "from red_agent.utils.aggregate import evaluate_all_rounds; evaluate_all_rounds()"

clean:
	rm -f logs/transcript.txt
	rm -f logs/transcript*.txt
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