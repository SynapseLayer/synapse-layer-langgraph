![Version](https://img.shields.io/badge/version-0.1.0-blue)
![PyPI](https://img.shields.io/pypi/v/synapse-layer)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-alpha-yellow)

# Synapse Layer × LangGraph

> **RAG retrieves. Synapse remembers.**

**Persistent memory infrastructure for AI agents — AES-256-GCM encrypted at rest, semantic search, MCP-native.**

LangGraph checkpoint saver backed by Synapse Layer. Persists agent state across sessions with AES-256-GCM encryption at rest and per-operation random IV.

---

## Install

```bash
pip install synapse-layer langgraph
```

## Usage

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from synapse_langgraph import SynapseCheckpointSaver

class State(TypedDict):
    messages: list[str]

def step(state: State) -> State:
    return {"messages": state["messages"] + ["done"]}

saver = SynapseCheckpointSaver(token=os.environ["SYNAPSE_TOKEN"])

graph = StateGraph(State)
graph.add_node("step", step)
graph.set_entry_point("step")
graph.add_edge("step", END)

app = graph.compile(checkpointer=saver)
result = app.invoke(
    {"messages": []},
    config={"configurable": {"thread_id": "session-1"}},
)
print(result)  # {"messages": ["done"]}

# Next invocation with same thread_id recovers state.
```

Get your token: [forge.synapselayer.org](https://forge.synapselayer.org) → Dashboard → Connect

---

## How It Works

```
LangGraph Agent
    │
    ├── put()      → serialize state → Synapse Layer (encrypted at rest)
    ├── get_tuple() → recall state    → deserialize → resume
    └── put_writes()→ no-op (MVP)

Session 1 ─── save ───┐
                       │ Synapse Layer (AES-256-GCM at rest)
Session 2 ─── recall ─┘
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Fail-closed on writes | `put()` raises `RuntimeError` — never silently lose state |
| Fail-open on reads | `get_tuple()` returns `None` — never break the agent |
| JSON serialization | Portable, inspectable. No pickle. |
| `put_writes()` is no-op | MVP scope — intermediate writes bundled in `put()` |
| Token via env var | Never hardcoded. `os.environ["SYNAPSE_TOKEN"]` |

---

## Related Projects

| Project | Description |
|---------|-------------|
| [synapse-layer](https://github.com/SynapseLayer/synapse-layer) | Core server — MCP endpoint, encryption engine, pgvector indexing |
| [synapse-sdk-python](https://github.com/SynapseLayer/synapse-sdk-python) | Python SDK — LangChain, CrewAI, and A2A protocol adapters |
| [synapse-layer-skill](https://github.com/SynapseLayer/synapse-layer-skill) | MCP skill configuration for Claude Desktop, Cursor, Windsurf |

---

## Links

- Website: [synapselayer.org](https://synapselayer.org)
- Forge (Dashboard): [forge.synapselayer.org](https://forge.synapselayer.org)
- PyPI: [pypi.org/project/synapse-layer](https://pypi.org/project/synapse-layer/)
- Core Repo: [github.com/SynapseLayer/synapse-layer](https://github.com/SynapseLayer/synapse-layer)
- MCP Endpoint: `https://forge.synapselayer.org/api/mcp`

Apache 2.0 © Synapse Layer
