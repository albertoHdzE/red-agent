from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from red_agent.agents.base import DebateAgent


class ArenaState(TypedDict):
    topic: str
    conversation: str
    active_agents: List[str]


def build_debate_graph(agents: List[DebateAgent]) -> StateGraph:
    state_graph = StateGraph(ArenaState)

    # Define each agent node
    for agent in agents:

        def make_node(agent_instance: DebateAgent):
            def node_fn(state: ArenaState) -> ArenaState:
                if agent_instance.name not in state["active_agents"]:
                    return state  # Skip if opted out

                output = agent_instance.generate_comment(
                    topic=state["topic"], conversation=state["conversation"]
                )

                if "Nothing to add" in output:
                    state["active_agents"].remove(agent_instance.name)
                else:
                    state["conversation"] += f"\n{output}"
                    agent_instance.memory.append(output)

                return state

            return node_fn

        state_graph.add_node(agent.name, make_node(agent))

    # Round-robin edges
    agent_names = [agent.name for agent in agents]
    for i, name in enumerate(agent_names):
        next_name = agent_names[i + 1] if i + 1 < len(agent_names) else END
        state_graph.add_edge(name, next_name)

    state_graph.set_entry_point(agent_names[0])
    return state_graph.compile()
