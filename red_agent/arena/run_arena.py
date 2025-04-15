import logging
import random
from pathlib import Path

from rich import print
from rich.panel import Panel

from red_agent.agents.base import DebateAgent
from red_agent.agents.referee import RefereeAgent
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
    logger.info("Starting debate arena")

    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration with {len(config['agents'])} agents")

    # 🎯 Pick a topic
    topic = random.choice(topics)
    logger.info(f"Selected topic: {topic}")
    print(
        Panel.fit(
            f"[bold yellow]🧠 Debate Topic:[/bold yellow]\n{topic}",
            title="TOPIC",
        )
    )

    # 🎭 Create agents from configuration
    agents = []
    for agent_config in config["agents"]:
        agent = DebateAgent(
            name=agent_config["name"],
            role=agent_config["role"],
            model=agent_config["model"],
            description=agent_config.get("description", ""),
            min_turns=config["debate"]["min_turns_per_agent"],
            max_turns=config["debate"]["max_turns_per_agent"],
        )
        agents.append(agent)

    logger.info(f"Created agents: {[agent.name for agent in agents]}")

    # Create referee with proper configuration
    referee = RefereeAgent(
        model="deepseek-coder-v2",  # Or use config value
        prompt_path=Path(__file__).parent.parent.parent
        / "red_agent/prompts/referee_csv_prompt.txt",
    )

    # 🧠 Build LangGraph
    logger.info("Building debate graph")
    graph = build_debate_graph(agents)

    # 🧠 Initial state
    state = {
        "topic": topic,
        "conversation": "",
        "active_agents": [agent.name for agent in agents],
        "current_agent_index": 0,
        "turn_counts": {agent.name: 0 for agent in agents},
        "config": config,
        # Remove referee from state since we're using a global instance
    }

    # Initialize logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous logs - but don't delete transcript.txt
    # if (logs_dir / "transcript.txt").exists():
    #     (logs_dir / "transcript.txt").unlink()  # Comment out this line to preserve transcript
    if (logs_dir / "evaluation.csv").exists():
        (logs_dir / "evaluation.csv").unlink()

    # Remove the agent-specific folder cleanup since we're not using them anymore

    # Debate loop
    round_count = 0

    # Set maximum rounds to prevent infinite loops
    max_rounds = len(agents) * config["debate"]["max_turns_per_agent"] + 1

    while state["active_agents"] and round_count < max_rounds:
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
            new_lines = (
                state["conversation"].replace(prev_conversation, "").strip()
            )
            for line in new_lines.split("\n"):
                if line.strip():
                    print(f"[white]{line.strip()}[/white]")

            # Generate evaluation data for this round
            comments = []
            for agent in agents:
                if (
                    agent.turn_count > 0
                ):  # Only evaluate agents who participated
                    comment = {
                        "character": agent.name,
                        "comment_number": agent.turn_count,
                        "Ethical Soundness": random.choice(
                            ["High", "Medium", "Low"]
                        ),
                        "Risk Assessment": random.choice(
                            ["Low", "Medium", "High"]
                        ),
                        "Alignment and Divergence": random.choice(
                            ["Aligned", "Neutral", "Divergent"]
                        ),
                        "Agent's role": agent.role,
                        "Sentiment analysis": random.choice(
                            ["Positive", "Neutral", "Negative"]
                        ),
                    }
                    comments.append(comment)

            # Have referee evaluate the comments using the prompt
            for comment in comments:
                referee.evaluate_with_prompt(comment)

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
        "\n[green]✅ Debate ended. Transcripts and evaluations saved.[/green]"
    )


if __name__ == "__main__":
    main()

# Remove the transcript check code that was outside the main function
