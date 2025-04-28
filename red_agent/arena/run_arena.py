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
        "topic_index": topic_index,
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
                        if agent_name in ["mistral-openorca", "tinyllama"]
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


def run_referee_evaluation(logs_dir, testing_mode, wide_mode, num_topics):
    referee = RefereeAgent()
    evaluation_csv_path = logs_dir / "evaluation.csv"
    print(
        "[yellow]🔍 Running referee evaluation on all transcripts...[/yellow]"
    )
    logger.info("Starting referee evaluation on all transcripts.")

    # Determine which transcript files to evaluate based on the mode
    if testing_mode or (wide_mode and num_topics == 1):
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
        if testing_mode or (wide_mode and num_topics == 1):
            topic_index = 0
            topic = topics[topic_index]
            logger.info(
                f"Evaluating transcript for testing/wide mode (single topic): {transcript_path}"
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
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Run in wide mode with specified number of topics, minimum comments, and number of agents",
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=1,
        help="Number of topics to debate in wide mode (default: 1)",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=3,
        help="Minimum number of comments per agent in wide mode (default: 3)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="Number of agents to participate in wide mode (1 to 5, default: 5)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration with {len(config['agents'])} agents")

    # Validate arguments for wide mode
    if args.wide:
        if args.num_agents < 1 or args.num_agents > len(config["agents"]):
            logger.error(
                f"Number of agents must be between 1 and {len(config['agents'])} in wide mode."
            )
            print(
                f"[red]Error: Number of agents must be between 1 and {len(config['agents'])} in wide mode.[/red]"
            )
            return
        if args.topics < 1:
            logger.error("Number of topics must be at least 1 in wide mode.")
            print(
                "[red]Error: Number of topics must be at least 1 in wide mode.[/red]"
            )
            return
        if args.min_comments < 1:
            logger.error(
                "Minimum comments per agent must be at least 1 in wide mode."
            )
            print(
                "[red]Error: Minimum comments per agent must be at least 1 in wide mode.[/red]"
            )
            return

    # Override testing_mode and wide_mode
    testing_mode = args.test
    wide_mode = args.wide
    config["debate"]["testing_mode"] = testing_mode
    logger.info(
        f"Running in {'testing' if testing_mode else 'wide' if wide_mode else 'real'} mode"
    )

    # Initialize logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create all agents from configuration
    all_agents = []
    all_agent_names = [
        agent_config["name"] for agent_config in config["agents"]
    ]
    for agent_config in config["agents"]:
        agent = DebateAgent(
            name=agent_config["name"],
            role=agent_config["role"],
            model=agent_config["model"],
            description=agent_config.get("description", ""),
            min_turns=config["debate"]["min_turns_per_agent"],
            max_turns=config["debate"]["max_turns_per_agent"],
            active_agent_names=all_agent_names,  # Pass all agent names initially
        )
        all_agents.append(agent)

    logger.info(f"Created all agents: {[agent.name for agent in all_agents]}")

    # Select agents for the debate
    if wide_mode:
        num_agents = args.num_agents
        selected_agents = random.sample(all_agents, num_agents)
        selected_agent_names = [agent.name for agent in selected_agents]
        # Update active_agent_names for selected agents
        for agent in selected_agents:
            agent.active_agent_names = selected_agent_names
            agent.min_turns = args.min_comments
        config["debate"]["min_turns_per_agent"] = args.min_comments
        logger.info(
            f"Selected {num_agents} agents for wide mode: {selected_agent_names}"
        )
    else:
        selected_agents = all_agents
        selected_agent_names = all_agent_names
        logger.info(
            f"Using all agents: {[agent.name for agent in selected_agents]}"
        )

    if testing_mode:
        # Testing mode: Run a single random topic with all agents
        topic = random.choice(topics)
        logger.info(f"Testing mode: Selected random topic: {topic}")
        run_single_topic(
            topic,
            topic_index=None,
            agents=selected_agents,
            config=config,
            logs_dir=logs_dir,
        )
    elif wide_mode:
        # Wide mode: Run the specified number of topics
        num_topics = min(args.topics, len(topics))
        selected_topics = random.sample(topics, num_topics)
        logger.info(
            f"Wide mode: Selected {num_topics} topics: {selected_topics}"
        )
        for topic_idx, topic in enumerate(selected_topics, 1):
            run_single_topic(
                topic,
                topic_index=None if num_topics == 1 else topic_idx,
                agents=selected_agents,
                config=config,
                logs_dir=logs_dir,
            )
    else:
        # Real mode: Run all topics
        logger.info("Real mode: Generating transcripts for all topics")
        for topic_idx, topic in enumerate(topics, 1):
            run_single_topic(
                topic,
                topic_index=topic_idx,
                agents=selected_agents,
                config=config,
                logs_dir=logs_dir,
            )

    # Run referee evaluation
    try:
        run_referee_evaluation(
            logs_dir,
            testing_mode=testing_mode,
            wide_mode=wide_mode,
            num_topics=num_topics if wide_mode else len(topics),
        )
    except Exception as e:
        print(f"[red]Error during referee evaluation: {e}[/red]")
        logger.error(f"Error during referee evaluation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
