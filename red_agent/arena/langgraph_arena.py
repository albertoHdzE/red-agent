import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from red_agent.agents.base import DebateAgent
from red_agent.utils.config import load_config

logger = logging.getLogger("red_agent.langgraph")


class ArenaState(TypedDict):
    topic: str
    conversation: str
    active_agents: List[str]
    current_agent_index: int
    turn_counts: Dict[str, int]
    config: Dict[str, Any]


def build_debate_graph(agents: List[DebateAgent]) -> StateGraph:
    logger.info(f"Building debate graph with {len(agents)} agents")
    state_graph = StateGraph(ArenaState)

    # Get agent names for easier reference
    agent_names = [agent.name for agent in agents]

    # Define each agent node
    for agent in agents:
        logger.info(f"Adding agent node: {agent.name}")

        def make_node(agent_instance: DebateAgent):
            def node_fn(state: ArenaState) -> ArenaState:
                logger.info(f"Executing node for agent: {agent_instance.name}")

                if agent_instance.name not in state["active_agents"]:
                    logger.info(
                        f"Agent {agent_instance.name} is no longer active, skipping"
                    )
                    return state  # Skip if opted out

                logger.info(f"Generating comment for {agent_instance.name}")

                # Ensure the agent has access to the full conversation
                full_conversation = state["conversation"]

                output = agent_instance.generate_comment(
                    topic=state["topic"], conversation=full_conversation
                )
                logger.info(
                    f"Generated output for {agent_instance.name}: {output[:50]}..."
                )

                # In the node_fn function
                try:
                    # Explicitly log to transcript here as well
                    logs_dir = Path(__file__).parent.parent.parent / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)

                    transcript_path = logs_dir / "transcript.txt"
                    logger.debug(
                        f"LangGraph writing to transcript at: {transcript_path.absolute()}"
                    )

                    with open(transcript_path, "a") as f:
                        f.write(f"{output}\n")

                    # Verify file exists and has content
                    if transcript_path.exists():
                        logger.debug(
                            f"LangGraph transcript file exists, size: {transcript_path.stat().st_size} bytes"
                        )
                    else:
                        logger.error(
                            "LangGraph transcript file does not exist after writing!"
                        )

                except Exception as e:
                    logger.error(
                        f"Error in LangGraph logging transcript: {str(e)}",
                        exc_info=True,
                    )

                # Update turn count
                state["turn_counts"][agent_instance.name] += 1
                current_turn = state["turn_counts"][agent_instance.name]

                # Check if agent should be removed (max turns or nothing to add)
                max_turns = state["config"]["debate"]["max_turns_per_agent"]
                if "Nothing to add" in output or current_turn >= max_turns:
                    logger.info(
                        f"Agent {agent_instance.name} has nothing to add or reached max turns, removing from active agents"
                    )
                    state["active_agents"].remove(agent_instance.name)
                else:
                    logger.info(
                        f"Adding {agent_instance.name}'s comment to conversation"
                    )
                    state["conversation"] += f"\n{output}"
                    agent_instance.memory.append(output)

                # Move to the next agent
                state["current_agent_index"] = (
                    state["current_agent_index"] + 1
                ) % len(agent_names)
                logger.info(
                    f"Active agents after {agent_instance.name}'s turn: {state['active_agents']}"
                )
                return state

            return node_fn

        state_graph.add_node(agent.name, make_node(agent))

    # Define router function to determine next agent or end
    def router(state: ArenaState) -> str:
        logger.info(
            f"Router called with active agents: {state['active_agents']}"
        )

        # End the graph if no active agents remain
        if not state["active_agents"]:
            logger.info("No active agents remain, ending graph")
            return END

        # Find the next active agent in rotation
        start_idx = state["current_agent_index"]
        for i in range(len(agent_names)):
            idx = (start_idx + i) % len(agent_names)
            next_agent = agent_names[idx]
            if next_agent in state["active_agents"]:
                logger.info(f"Router selected next agent: {next_agent}")
                state["current_agent_index"] = idx
                return next_agent

        # If we get here, no active agents were found
        logger.info("No active agents found in rotation, ending graph")
        return END

    # Add conditional edges from each agent
    for agent in agents:
        state_graph.add_conditional_edges(
            agent.name,
            router,
            {agent.name: agent.name for agent in agents} | {END: END},
        )

    # Get entry point from config
    config = load_config()
    entry_point = config["debate"]["entry_point"]
    if entry_point not in agent_names:
        logger.warning(
            f"Entry point {entry_point} not found in agents, using first agent"
        )
        entry_point = agent_names[0]

    logger.info(f"Setting entry point to {entry_point}")
    state_graph.set_entry_point(entry_point)

    return state_graph.compile()
