import argparse
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


def run_single_topic(topic, topic_index, agents, config, logs_dir):
    """
    Run a debate for a single topic and save the transcript.

    Args:
        topic (str): The topic to debate.
        topic_index (int or None): The index of the topic (1-based) for naming the transcript file.
                                   If None, saves to transcript.txt (testing mode).
        agents (list): List of DebateAgent instances.
        config (dict): Configuration dictionary.
        logs_dir (Path): Directory to save the transcript.
    """
    logger.info(f"Processing topic: {topic}")
    print(
        Panel.fit(
            f"[bold yellow]🧠 Debate Topic{' ' + str(topic_index) if topic_index is not None else ''}:[/bold yellow]\n{topic}",
            title=f"TOPIC{(' ' + str(topic_index)) if topic_index is not None else ''}",
        )
    )

    # Determine transcript path
    if topic_index is not None:
        transcript_path = logs_dir / f"transcript_{topic_index}.txt"
    else:
        transcript_path = logs_dir / "transcript.txt"

    # Clear the transcript file if it exists
    if transcript_path.exists():
        transcript_path.unlink()
        logger.info(f"Removed existing {transcript_path} for topic.")

    # Build LangGraph
    logger.info("Building debate graph")
    graph = build_debate_graph(agents)

    # Reset agent states
    for agent in agents:
        agent.memory = []
        agent.finished = False
        agent.turn_count = 0

    # Initial state
    state = {
        "topic": topic,
        "conversation": "",
        "active_agents": [agent.name for agent in agents],
        "current_agent_index": 0,
        "turn_counts": {agent.name: 0 for agent in agents},
        "config": config,
        "topic_index": topic_index,  # Pass topic index for transcript naming
    }

    # Debate loop
    round_count = 0
    while len(state["active_agents"]) > 1:
        round_count += 1
        print(
            Panel.fit(
                f"🔁 Round {round_count}\nActive agents: [bold]{', '.join(state['active_agents'])}[/bold]",
                title=f"{'Topic ' + str(topic_index) + ' - ' if topic_index is not None else ''}Round {round_count}",
                border_style="white",
                padding=(0, 1),
                width=40,
            )
        )

        try:
            logger.info(
                f"Invoking graph for {'topic ' + str(topic_index) + ', ' if topic_index is not None else ''}round {round_count}"
            )
            prev_conversation = state["conversation"]
            state = graph.invoke(state)

            # Print agent outputs with red agent highlighting
            logger.info("PRINTING COMMENTS FROM AGENTS")
            new_lines = (
                state["conversation"].replace(prev_conversation, "").strip()
            )
            for line in new_lines.split("\n"):
                if line.strip():
                    agent_name = line.split(":")[0]
                    color = (
                        "red"
                        if agent_name in ["Nemesis", "Chaos"]
                        else "white"
                    )
                    print(f"[{color}]{line.strip()}[/{color}]")

        except Exception as e:
            logger.error(
                f"Error in debate loop for {'topic ' + str(topic_index) if topic_index is not None else 'topic'}: {str(e)}",
                exc_info=True,
            )
            print(
                f"[red]Error in debate for {'topic ' + str(topic_index) if topic_index is not None else 'topic'}: {str(e)}[/red]"
            )
            break

    # Check transcript file
    try:
        if transcript_path.exists():
            with open(transcript_path, "r") as f:
                content = f.read()
            print(
                f"\n[blue]Transcript {transcript_path.name} saved with {len(content)} bytes of content[/blue]"
            )
        else:
            print(
                f"\n[red]Warning: Transcript file {transcript_path.name} does not exist![/red]"
            )
    except Exception as e:
        print(
            f"\n[red]Error checking transcript {transcript_path.name}: {str(e)}[/red]"
        )

    print(
        f"\n[green]✅ Debate for {'topic ' + str(topic_index) if topic_index is not None else 'topic'} ended. Transcript saved.[/green]"
    )


def run_referee_evaluation(logs_dir, testing_mode):
    referee = RefereeAgent()
    evaluation_csv_path = logs_dir / "evaluation.csv"
    print(
        "[yellow]🔍 Running referee evaluation on all transcripts...[/yellow]"
    )
    logger.info("Starting referee evaluation on all transcripts.")

    # Determine which transcript files to evaluate based on the mode
    if testing_mode:
        transcript_files = [logs_dir / "transcript.txt"]
    else:
        transcript_files = sorted(
            logs_dir.glob("transcript_*.txt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    if not transcript_files:
        logger.error("No transcript files found in logs directory.")
        print("[red]Error: No transcript files found in logs directory.[/red]")
        return

    # Clear the evaluation CSV if it exists to start fresh
    if evaluation_csv_path.exists():
        evaluation_csv_path.unlink()
        logger.info(f"Removed existing {evaluation_csv_path} to start fresh.")

    for transcript_path in transcript_files:
        # Extract topic index based on filename
        if testing_mode:
            topic_index = 0
            topic = topics[topic_index]
            logger.info(
                f"Evaluating transcript for testing mode: {transcript_path}"
            )
        else:
            topic_index = int(transcript_path.stem.split("_")[1]) - 1
            topic = topics[topic_index]
            logger.info(
                f"Evaluating transcript for topic {topic_index + 1}: {topic}"
            )

        # Verify transcript exists and is not empty
        if not transcript_path.exists():
            logger.error(f"Transcript file {transcript_path} does not exist.")
            print(
                f"[red]Error: Transcript file {transcript_path} does not exist.[/red]"
            )
            continue

        with open(transcript_path, "r") as f:
            transcript_content = f.read()

        if not transcript_content.strip():
            logger.error(f"Transcript file {transcript_path} is empty.")
            print(
                f"[red]Error: Transcript file {transcript_path} is empty.[/red]"
            )
            continue

        # Run the evaluation with the topic
        try:
            referee.evaluate_transcript(
                transcript_path, evaluation_csv_path, topic
            )
            print(
                f"[green]✅ Evaluated transcript {transcript_path}. Results appended to {evaluation_csv_path}[/green]"
            )
            logger.info(f"Referee evaluation complete for {transcript_path}.")
        except Exception as e:
            logger.error(
                f"Error during referee evaluation of {transcript_path}: {str(e)}",
                exc_info=True,
            )
            print(
                f"[red]Error during referee evaluation of {transcript_path}: {str(e)}[/red]"
            )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the debate arena.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with a random topic",
    )
    args = parser.parse_args()

    # Load configuration from agents.yaml where agents are defined
    config = load_config()
    logger.info(f"Loaded configuration with {len(config['agents'])} agents")

    # Override testing_mode based on command-line argument
    testing_mode = args.test
    config["debate"]["testing_mode"] = testing_mode
    logger.info(f"Running in {'testing' if testing_mode else 'real'} mode")

    # Initialize logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create agents from configuration
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

    if testing_mode:
        # Testing mode: Run a single random topic
        topic = random.choice(topics)
        logger.info(f"Testing mode: Selected random topic: {topic}")
        run_single_topic(
            topic,
            topic_index=None,
            agents=agents,
            config=config,
            logs_dir=logs_dir,
        )
    else:
        # Real mode: Run all topics
        logger.info("Real mode: Generating transcripts for all topics")
        for topic_idx, topic in enumerate(
            topics, 1
        ):  # 1-based index for filenames
            run_single_topic(
                topic,
                topic_index=topic_idx,
                agents=agents,
                config=config,
                logs_dir=logs_dir,
            )

    # After all transcripts are generated, run referee evaluation
    try:
        run_referee_evaluation(logs_dir, testing_mode=testing_mode)
    except Exception as e:
        print(f"[red]Error during referee evaluation: {e}[/red]")
        logger.error(f"Error during referee evaluation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
