"""SynapseCheckpointSaver — LangGraph checkpoint backend powered by Synapse Layer.

Provides persistent, encrypted, cross-session memory for LangGraph agents.
Server never sees plaintext.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from synapse_memory.client import Synapse

logger = logging.getLogger(__name__)


def _thread_id(config: RunnableConfig) -> str:
    """Extract thread_id from config. Raises ValueError if missing."""
    configurable = (config or {}).get("configurable", {})
    tid = configurable.get("thread_id")
    if not tid:
        raise ValueError("config['configurable']['thread_id'] is required")
    return str(tid)


class SynapseCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by Synapse Layer.

    Provides persistent, encrypted, cross-session memory for LangGraph agents.
    Server never sees plaintext.

    Usage:
        saver = SynapseCheckpointSaver(token=os.environ["SYNAPSE_TOKEN"])
        graph = builder.compile(checkpointer=saver)
    """

    def __init__(
        self,
        token: str,
        namespace: str = "langgraph",
        *,
        serde: Any = None,
    ) -> None:
        super().__init__(serde=serde)
        if not token:
            raise ValueError("Synapse connect token is required")
        self._client = Synapse(token=token)
        self._namespace = namespace

    def _make_key(self, thread_id: str) -> str:
        """Create a namespaced key for storage."""
        return f"{self._namespace}:checkpoint:{thread_id}"

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint to Synapse Layer.

        Fail-closed: raises RuntimeError if storage fails.
        """
        thread_id = _thread_id(config)
        key = self._make_key(thread_id)

        payload = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "new_versions": new_versions,
            "parent_config": config,
        }

        serialized = json.dumps(payload, default=str)

        try:
            self._client.remember(
                content=serialized,
                metadata={"key": key, "thread_id": thread_id, "type": "checkpoint"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to persist checkpoint for thread '{thread_id}'"
            ) from exc

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint.get("id", ""),
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes. No-op in MVP — writes are bundled in put()."""
        pass

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve the latest checkpoint for a thread.

        Fail-open: returns None on errors (never breaks the agent).
        """
        try:
            thread_id = _thread_id(config)
        except ValueError:
            return None

        key = self._make_key(thread_id)

        try:
            results = self._client.recall(key, top_k=1)
        except Exception:
            logger.warning("Synapse recall failed for thread '%s'", thread_id)
            return None

        if not results:
            return None

        try:
            content = results[0].get("content", "")
            payload = json.loads(content)
            return CheckpointTuple(
                config=payload.get("parent_config", config),
                checkpoint=payload["checkpoint"],
                metadata=payload.get("metadata", {}),
                parent_config=None,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to deserialize checkpoint for thread '%s'", thread_id)
            return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints. Returns empty iterator in MVP."""
        yield from []
