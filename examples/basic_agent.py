"""Basic LangGraph agent with Synapse Layer persistent memory.

Requirements:
    pip install synapse-layer langgraph

Usage:
    export SYNAPSE_TOKEN=sk_connect_YOUR_TOKEN
    python basic_agent.py
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from synapse_langgraph import SynapseCheckpointSaver


class AgentState(TypedDict):
    messages: list[str]
    step_count: int


def process_step(state: AgentState) -> AgentState:
    """Simple processing node that increments the step counter."""
    count = state.get("step_count", 0) + 1
    messages = state.get("messages", []) + [f"Step {count} completed"]
    return {"messages": messages, "step_count": count}


def main():
    token = os.environ.get("SYNAPSE_TOKEN")
    if not token:
        print("Set SYNAPSE_TOKEN env var first.")
        print("Get yours at: synapselayer.org -> Dashboard -> Connect")
        return

    # 1. Create Synapse-backed checkpoint saver
    saver = SynapseCheckpointSaver(token=token)

    # 2. Build a simple graph
    builder = StateGraph(AgentState)
    builder.add_node("process", process_step)
    builder.set_entry_point("process")
    builder.add_edge("process", END)

    graph = builder.compile(checkpointer=saver)

    # 3. First invocation — state is created
    config = {"configurable": {"thread_id": "demo-session-1"}}
    result1 = graph.invoke(
        {"messages": [], "step_count": 0},
        config=config,
    )
    print(f"Run 1: {result1}")

    # 4. Second invocation — state persists across calls
    result2 = graph.invoke(
        {"messages": [], "step_count": 0},
        config=config,
    )
    print(f"Run 2: {result2}")
    print("Memory persisted across invocations \u2713")


if __name__ == "__main__":
    main()
