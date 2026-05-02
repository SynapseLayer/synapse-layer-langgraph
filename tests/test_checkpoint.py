"""Unit tests for SynapseCheckpointSaver.

All tests mock the Synapse SDK — no real API calls.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from synapse_langgraph.checkpoint import SynapseCheckpointSaver


# ---- Fixtures ----


def _make_config(thread_id: str = "test-thread"):
    return {"configurable": {"thread_id": thread_id}}


def _make_checkpoint():
    return {"id": "ckpt-001", "ts": "2026-01-01T00:00:00Z", "channel_values": {"messages": ["hello"]}}


def _make_metadata():
    return {"source": "input", "step": 1}


# ---- TEST 1: put() calls client.remember() correctly ----


@patch("synapse_langgraph.checkpoint.Synapse")
def test_put_calls_remember(mock_synapse_cls):
    mock_client = MagicMock()
    mock_synapse_cls.return_value = mock_client

    saver = SynapseCheckpointSaver(token="sk_connect_YOUR_TOKEN")
    config = _make_config()
    checkpoint = _make_checkpoint()
    metadata = _make_metadata()

    result = saver.put(config, checkpoint, metadata, {})

    mock_client.remember.assert_called_once()
    call_kwargs = mock_client.remember.call_args
    content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content") or call_kwargs[0][0]
    parsed = json.loads(content)
    assert parsed["checkpoint"]["id"] == "ckpt-001"
    assert result["configurable"]["thread_id"] == "test-thread"


# ---- TEST 2: get_tuple() returns None on empty recall ----


@patch("synapse_langgraph.checkpoint.Synapse")
def test_get_tuple_returns_none_on_empty(mock_synapse_cls):
    mock_client = MagicMock()
    mock_client.recall.return_value = []
    mock_synapse_cls.return_value = mock_client

    saver = SynapseCheckpointSaver(token="sk_connect_YOUR_TOKEN")
    result = saver.get_tuple(_make_config())

    assert result is None


# ---- TEST 3: get_tuple() deserializes correctly ----


@patch("synapse_langgraph.checkpoint.Synapse")
def test_get_tuple_deserializes(mock_synapse_cls):
    mock_client = MagicMock()
    checkpoint = _make_checkpoint()
    metadata = _make_metadata()
    payload = {
        "checkpoint": checkpoint,
        "metadata": metadata,
        "new_versions": {},
        "parent_config": _make_config(),
    }
    mock_client.recall.return_value = [{"content": json.dumps(payload)}]
    mock_synapse_cls.return_value = mock_client

    saver = SynapseCheckpointSaver(token="sk_connect_YOUR_TOKEN")
    result = saver.get_tuple(_make_config())

    assert result is not None
    assert result.checkpoint["id"] == "ckpt-001"
    assert result.metadata["source"] == "input"


# ---- TEST 4: put() raises RuntimeError on failure ----


@patch("synapse_langgraph.checkpoint.Synapse")
def test_put_raises_on_failure(mock_synapse_cls):
    mock_client = MagicMock()
    mock_client.remember.side_effect = ConnectionError("API down")
    mock_synapse_cls.return_value = mock_client

    saver = SynapseCheckpointSaver(token="sk_connect_YOUR_TOKEN")

    with pytest.raises(RuntimeError, match="Failed to persist checkpoint"):
        saver.put(_make_config(), _make_checkpoint(), _make_metadata(), {})


# ---- TEST 5: No logs contain token or sensitive content ----


@patch("synapse_langgraph.checkpoint.Synapse")
def test_no_sensitive_data_in_logs(mock_synapse_cls, caplog):
    mock_client = MagicMock()
    mock_client.recall.side_effect = ConnectionError("fail")
    mock_synapse_cls.return_value = mock_client

    saver = SynapseCheckpointSaver(token="sk_connect_YOUR_TOKEN")

    with caplog.at_level(logging.WARNING):
        result = saver.get_tuple(_make_config())

    assert result is None
    log_text = caplog.text.lower()
    assert "sk_connect" not in log_text
    assert "token" not in log_text
    assert "content" not in log_text or "checkpoint" not in log_text
