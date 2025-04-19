import logging
import random
from pathlib import Path

from rich import print
from rich.panel import Panel

from red_agent.agents.base import DebateAgent

# Removed RefereeAgent import
from red_agent.arena.langgraph_arena import build_debate_graph
from red_agent.data.topics import topics
from red_agent.utils.config import load_config

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger("red_agent.arena")


def main():
    logger.info("=========================")
    logger.info("Starting debate arena")
    logger.info("=========================")

    # Load configuration from agents.yaml where agents are defined
    config = load_config()
    logger.info(f"Loaded configuration with {len(config['agents'])} agents")

    # Pick a topic randomly
    topic = random.choice(topics)
    logger.info(f"Selected topic: {topic}")
    print(
        Panel.fit(
            f"[bold yellow]🧠 Debate Topic:[/bold yellow]\n{topic}",
            title="TOPIC",
        )
    )

    # Create agents from configuration
    agents = []
    for agent_config in config["agents"]:
        # from agents.base
        agent = DebateAgent(
            name=agent_config["name"],
            role=agent_config["role"],
            model=agent_config["model"],
            description=agent_config.get("description", ""),
            # Removed max_turns parameter
        )
        agents.append(agent)

    logger.info(f"Created agents: {[agent.name for agent in agents]}")

    # Build LangGraph
    logger.info("Building debate graph")
    graph = build_debate_graph(agents)

    # Initial state
    state = {
        "topic": topic,
        "conversation": "",
        "active_agents": [agent.name for agent in agents],
        "current_agent_index": 0,
        "turn_counts": {agent.name: 0 for agent in agents},
        "config": config,
    }

    # Initialize logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Debate loop
    round_count = 0

    # Remove max_rounds logic since agents will stop when they say "Nothing to add"
    while len(state["active_agents"]) > 1:
        round_count += 1
        print(
            Panel.fit(
                f"🔁 Round {round_count}\nActive agents: [bold]{', '.join(state['active_agents'])}[/bold]",
                title=f"Round {round_count}",
                border_style="white",
                padding=(0, 1),
                width=40,
            )
        )

        try:
            logger.info(f"Invoking graph for round {round_count}")
            prev_conversation = state["conversation"]
            state = graph.invoke(state)

            # Print agent outputs
            logger.info("PRINTING COMMENTS FROM AGENTS")
            new_lines = (
                state["conversation"].replace(prev_conversation, "").strip()
            )
            for line in new_lines.split("\n"):
                if line.strip():
                    print(f"[white]{line.strip()}[/white]")

        except Exception as e:
            logger.error(f"Error in debate loop: {str(e)}", exc_info=True)
            print(f"[red]Error in debate: {str(e)}[/red]")
            break

    # Run evaluations
    # After the evaluate_all_rounds() call

    # Check transcript file
    try:
        transcript_path = logs_dir / "transcript.txt"
        if transcript_path.exists():
            with open(transcript_path, "r") as f:
                content = f.read()
            print(
                f"\n[blue]Transcript saved with {len(content)} bytes of content[/blue]"
            )
        else:
            print("\n[red]Warning: Transcript file does not exist![/red]")
    except Exception as e:
        print(f"\n[red]Error checking transcript: {str(e)}[/red]")

    print(
        "\n[green]✅ Debate ended. Transcript saved.[/green]"
    )  # Updated message


if __name__ == "__main__":
    main()
