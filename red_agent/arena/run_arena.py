import random
from pathlib import Path

from rich import print
from rich.panel import Panel

from red_agent.agents.base import DebateAgent
from red_agent.arena.langgraph_arena import build_debate_graph
from red_agent.data.topics import topics
from red_agent.utils.aggregate import evaluate_all_rounds


def main():
    # 🎯 Pick a topic
    topic = random.choice(topics)
    print(
        Panel.fit(
            f"[bold yellow]🧠 Debate Topic:[/bold yellow]\n{topic}",
            title="TOPIC",
        )
    )

    # 🎭 Define your agents
    agents = [
        DebateAgent(name="Athena", role="truth-seeker", model="mistral"),
        DebateAgent(name="Prometheus", role="radical-innovator", model="mistral"),
    ]

    # 🧠 Build LangGraph
    graph = build_debate_graph(agents)

    # 🧠 Initial state
    state = {
        "topic": topic,
        "conversation": "",
        "active_agents": [agent.name for agent in agents],
    }

    round_count = 0
    conversation = ""

    while state["active_agents"]:
        round_count += 1
        print(
            Panel.fit(
                f"[cyan]🔁 Round {round_count}[/cyan]\nActive agents: [bold]{', '.join(state['active_agents'])}[/bold]",
                title=f"Round {round_count}",
            )
        )

        state = graph.invoke(state)

        # Print agent outputs
        new_lines = state["conversation"].replace(conversation, "").strip()
        conversation = state["conversation"]
        for line in new_lines.split("\n"):
            if line.strip():
                print(f"[white]{line.strip()}[/white]")

        # 📝 Save per-round transcript
        round_path = Path(f"logs/round_{round_count}")
        round_path.mkdir(parents=True, exist_ok=True)
        with open(round_path / "transcript.txt", "w") as f:
            f.write(new_lines)

    print("\n[green]✅ Debate ended. Saving evaluations...[/green]")
    evaluate_all_rounds()


if __name__ == "__main__":
    main()
