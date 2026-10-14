"""Behavioral tests for manual-merge source ordering."""

from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from testing.import_isolation import load_module_fresh, stub_modules

_BACKEND = Path(__file__).resolve().parents[2]
_MISSING = object()
_MORNING = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
_NOON = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)


def _source(
    conversation_id: str,
    *,
    created_at=_MISSING,
    started_at=_MISSING,
    finished_at=_MISSING,
    transcript_segments=None,
) -> dict:
    conversation = {
        "id": conversation_id,
        "status": "completed",
        "language": "en",
        "source": "omi",
        "transcript_segments": transcript_segments or [],
    }
    for field, value in (
        ("created_at", created_at),
        ("started_at", started_at),
        ("finished_at", finished_at),
    ):
        if value is not _MISSING:
            conversation[field] = value
    return conversation


def _segment(text: str) -> dict:
    return {
        "text": text,
        "speaker": "SPEAKER_00",
        "speaker_id": 0,
        "is_user": False,
        "start": 0.0,
        "end": 5.0,
    }


@pytest.fixture(scope="module")
def merge_harness():
    """Load the real merge pipeline and model around fake external boundaries."""
    state = {
        "sources": {},
        "created": [],
        "completed": [],
        "deleted": [],
        "events": [],
    }

    database_package = ModuleType("database")
    database_package.__path__ = []  # type: ignore[attr-defined]

    def get_conversation(_uid, conversation_id):
        return state["sources"].get(conversation_id)

    def delete_conversation(_uid, conversation_id):
        state["deleted"].append(conversation_id)
        state["events"].append(("source_deleted", conversation_id))

    conversations_module = ModuleType("database.conversations")
    conversations_module.get_conversation = get_conversation
    conversations_module.get_conversation_photos = lambda _uid, _conversation_id: []
    conversations_module.store_conversation_photos = lambda *_args: None
    conversations_module.delete_conversation_photos = lambda *_args: None
    conversations_module.delete_conversation = delete_conversation
    conversations_module._delete_conversation_search_index = lambda *_args: None

    client_module = ModuleType("database._client")
    client_module.db = object()

    tombstones_module = ModuleType("database.conversation_tombstones")
    tombstones_module.db = object()
    tombstones_module.COLLECTION = "deleted_conversations"
    tombstones_module.is_deleted = lambda *_args: False
    tombstones_module.record_deletion = lambda *_args: None

    vector_module = ModuleType("database.vector_db")
    vector_module.delete_vector = lambda _uid, conversation_id: None

    action_items_module = ModuleType("database.action_items")
    action_items_module.get_action_items_by_conversation = lambda *_args: []
    action_items_module.delete_action_items_for_conversation = lambda *_args: None

    def save_conversation(uid, payload):
        state["created"].append((uid, copy.deepcopy(payload)))
        state["events"].append(("saved", payload["id"]))

    def complete_conversation(uid, conversation_id):
        state["completed"].append((uid, conversation_id))
        state["events"].append(("completed", conversation_id))

    lifecycle_module = ModuleType("utils.conversations.lifecycle")
    lifecycle_module.create_processing_conversation = save_conversation
    lifecycle_module.complete = complete_conversation

    class MemoryService:
        def __init__(self, db_client=None):
            self.db_client = db_client

        def retract_conversation_memories(self, _uid, conversation_id, *, on_authoritative_commit=None, **_kwargs):
            state["events"].append(("memories_retracted", conversation_id))
            if on_authoritative_commit is not None:
                on_authoritative_commit()

    memory_service_module = ModuleType("utils.memory.memory_service")
    memory_service_module.MemoryService = MemoryService

    retraction_module = ModuleType("utils.memory.retraction_scope")
    retraction_module.canonical_intake_is_fenced = lambda: False
    retraction_module.historical_source_conversation_ids = lambda *_args, **_kwargs: set()
    retraction_module.retraction_can_be_skipped = lambda *_args, **_kwargs: False

    storage_module = ModuleType("utils.other.storage")
    storage_module.compute_audio_files_fingerprint = lambda *_args: "fingerprint"
    storage_module.delete_conversation_audio_files = lambda _uid, conversation_id: state["events"].append(
        ("audio_deleted", conversation_id)
    )
    storage_module.enqueue_conversation_artifact_build = lambda *_args, **_kwargs: None
    storage_module.list_audio_chunks = lambda _uid, _conversation_id: []
    storage_module._get_storage_client = lambda: SimpleNamespace(bucket=lambda _name: object())
    storage_module.private_cloud_sync_bucket = "test-bucket"
    storage_module.owner_storage_write_gate = lambda *_args, **_kwargs: nullcontext()

    audit_module = ModuleType("utils.conversations.smart_merge_audit")
    audit_module.record_survivor_deleted = lambda *_args: None

    capture_module = ModuleType("utils.conversations.capture_shadow_outcomes")
    capture_module.record_capture_outcome = lambda *_args: None

    cloud_tasks_module = ModuleType("utils.cloud_tasks")
    cloud_tasks_module.is_audio_merge_dispatch_enabled = lambda: False

    process_module = ModuleType("utils.conversations.process_conversation")
    process_module.process_conversation = lambda *_args, **_kwargs: None

    notifications_module = ModuleType("utils.notifications")
    notifications_module.send_merge_completed_message = lambda *_args: None

    fakes = {
        "database": database_package,
        "database._client": client_module,
        "database.conversations": conversations_module,
        "database.conversation_tombstones": tombstones_module,
        "database.vector_db": vector_module,
        "database.action_items": action_items_module,
        "database.redis_db": ModuleType("database.redis_db"),
        "database.users": ModuleType("database.users"),
        "utils.cloud_tasks": cloud_tasks_module,
        "utils.conversations.lifecycle": lifecycle_module,
        "utils.conversations.smart_merge_audit": audit_module,
        "utils.conversations.capture_shadow_outcomes": capture_module,
        "utils.conversations.process_conversation": process_module,
        "utils.memory.memory_service": memory_service_module,
        "utils.memory.retraction_scope": retraction_module,
        "utils.notifications": notifications_module,
        "utils.other.storage": storage_module,
    }

    with stub_modules(fakes):
        merge = load_module_fresh(
            "utils.conversations.merge_conversations",
            str(_BACKEND / "utils" / "conversations" / "merge_conversations.py"),
        )
        yield SimpleNamespace(module=merge, state=state)


def _set_sources(merge_harness, sources):
    state = merge_harness.state
    state["sources"] = {source["id"]: source for source in sources}
    for name in ("created", "completed", "deleted", "events"):
        state[name].clear()


def _run_merge(merge_harness, sources, selected_ids):
    _set_sources(merge_harness, sources)
    merge_harness.module.perform_merge_async("test-uid", selected_ids, reprocess=False)


def _assert_successful_merge(merge_harness, caplog, expected_deleted, selected_ids):
    state = merge_harness.state
    assert len(state["created"]) == 1
    uid, payload = state["created"][0]
    assert uid == "test-uid"
    assert state["completed"] == [("test-uid", payload["id"])]
    assert len(state["deleted"]) == len(expected_deleted)
    assert set(state["deleted"]) == set(expected_deleted)
    assert payload["external_data"]["merge_metadata"]["source_conversation_ids"] == selected_ids

    event_positions = {event: index for index, event in enumerate(state["events"])}
    assert event_positions[("saved", payload["id"])] < event_positions[("completed", payload["id"])]
    for conversation_id in expected_deleted:
        assert event_positions[("completed", payload["id"])] < event_positions[("source_deleted", conversation_id)]

    failures = [
        record.getMessage()
        for record in caplog.records
        if record.name == "utils.conversations.merge_conversations"
        and ("Merge failed" in record.getMessage() or "Merge aborted" in record.getMessage())
    ]
    assert failures == []
    return payload


def _contains_year_one_timestamp(value) -> bool:
    if isinstance(value, datetime):
        return value.year == 1
    if isinstance(value, str):
        return value.startswith("0001-01-01")
    if isinstance(value, dict):
        return any(_contains_year_one_timestamp(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_year_one_timestamp(item) for item in value)
    return False


@pytest.mark.parametrize(
    ("noon_started_at", "noon_created_at"),
    [
        pytest.param(_MISSING, "2026-05-30T12:00:00Z", id="absent-start-iso-z-created"),
        pytest.param(None, "2026-05-30T14:00:00+02:00", id="none-start-offset-created"),
        pytest.param("invalid", datetime(2026, 5, 30, 12, 0), id="invalid-start-naive-created"),
        pytest.param(123, datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc), id="unsupported-start-aware-created"),
        pytest.param("2026-05-30T12:00:00Z", "2026-05-30T09:00:00Z", id="iso-z-start-precedes-created"),
        pytest.param("2026-05-30T14:00:00+02:00", "2026-05-30T09:00:00Z", id="offset-start-precedes-created"),
        pytest.param(datetime(2026, 5, 30, 12, 0), "2026-05-30T09:00:00Z", id="naive-start-precedes-created"),
        pytest.param(
            datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
            "2026-05-30T09:00:00Z",
            id="aware-start-precedes-created",
        ),
    ],
)
def test_merge_orders_sources_by_first_valid_timestamp(merge_harness, caplog, noon_started_at, noon_created_at):
    morning = _source(
        "morning",
        created_at=_MORNING,
        started_at=_MORNING,
        finished_at=datetime(2026, 5, 30, 10, 5, tzinfo=timezone.utc),
    )
    noon = _source(
        "noon",
        created_at=noon_created_at,
        started_at=noon_started_at,
        finished_at=datetime(2026, 5, 30, 12, 5, tzinfo=timezone.utc),
    )

    _run_merge(merge_harness, [noon, morning], ["noon", "morning"])

    payload = _assert_successful_merge(merge_harness, caplog, ["morning", "noon"], ["noon", "morning"])
    assert payload["created_at"] == _MORNING
    assert [detail["id"] for detail in payload["external_data"]["merge_metadata"]["source_details"]] == [
        "morning",
        "noon",
    ]


def test_merge_preserves_sequential_offsets_for_missing_start(merge_harness, caplog):
    morning = _source(
        "morning",
        created_at=_MORNING,
        started_at=_MORNING,
        finished_at=datetime(2026, 5, 30, 10, 0, 5, tzinfo=timezone.utc),
        transcript_segments=[_segment("morning segment")],
    )
    noon = _source(
        "noon",
        created_at=_NOON,
        transcript_segments=[_segment("noon segment")],
    )
    original_sources = copy.deepcopy([noon, morning])

    _run_merge(merge_harness, [noon, morning], ["noon", "morning"])
    assert [noon, morning] == original_sources

    payload = _assert_successful_merge(merge_harness, caplog, ["morning", "noon"], ["noon", "morning"])
    assert [segment["text"] for segment in payload["transcript_segments"]] == [
        "morning segment",
        "noon segment",
    ]
    assert [(segment["start"], segment["end"]) for segment in payload["transcript_segments"]] == [
        (0.0, 5.0),
        (5.0, 10.0),
    ]
    assert payload["created_at"] == _MORNING
    assert [detail["id"] for detail in payload["external_data"]["merge_metadata"]["source_details"]] == [
        "morning",
        "noon",
    ]


@pytest.mark.parametrize(
    "unknown_b_created_at",
    [
        pytest.param(123, id="unsupported-created"),
        pytest.param(None, id="none-created"),
        pytest.param("invalid", id="invalid-created"),
    ],
)
def test_merge_keeps_unknown_and_equal_timestamp_sources_stable(merge_harness, caplog, unknown_b_created_at):
    unknown_a = _source("unknown-a")
    unknown_b = _source("unknown-b", created_at=unknown_b_created_at, started_at="invalid")
    known = _source("known", created_at=_MORNING, started_at=_MORNING)

    _run_merge(merge_harness, [known, unknown_a, unknown_b], ["known", "unknown-a", "unknown-b"])

    payload = _assert_successful_merge(
        merge_harness,
        caplog,
        ["unknown-a", "unknown-b", "known"],
        ["known", "unknown-a", "unknown-b"],
    )
    assert [detail["id"] for detail in payload["external_data"]["merge_metadata"]["source_details"]] == [
        "unknown-a",
        "unknown-b",
        "known",
    ]
    assert payload["started_at"] is None
    assert not _contains_year_one_timestamp(payload)


@pytest.mark.parametrize(
    ("selected_ids", "expected_ids"),
    [
        pytest.param(["started", "created-only"], ["started", "created-only"], id="started-source-first"),
        pytest.param(["created-only", "started"], ["created-only", "started"], id="created-source-first"),
    ],
)
def test_merge_keeps_equal_effective_timestamps_in_input_order(merge_harness, caplog, selected_ids, expected_ids):
    started = _source(
        "started",
        created_at="2026-05-30T09:00:00Z",
        started_at=_NOON,
    )
    created_only = _source("created-only", created_at="2026-05-30T12:00:00Z")

    _run_merge(merge_harness, [started, created_only], selected_ids)

    payload = _assert_successful_merge(merge_harness, caplog, expected_ids, selected_ids)
    assert [detail["id"] for detail in payload["external_data"]["merge_metadata"]["source_details"]] == expected_ids
    assert not _contains_year_one_timestamp(payload)


@pytest.mark.parametrize(
    "noon_started_at",
    [
        pytest.param(None, id="none-start"),
        pytest.param(_MISSING, id="absent-start"),
        pytest.param("invalid", id="invalid-start"),
        pytest.param(123, id="unsupported-start"),
    ],
)
def test_validation_orders_startless_source_before_later_gap(merge_harness, noon_started_at):
    morning = _source(
        "morning",
        created_at="2026-05-30T10:00:00Z",
        started_at="2026-05-30T10:00:00Z",
        finished_at="2026-05-30T10:30:00Z",
    )
    noon = _source(
        "noon",
        created_at="2026-05-30T12:00:00Z",
        started_at=noon_started_at,
        finished_at="2026-05-30T12:30:00Z",
    )
    afternoon = _source(
        "afternoon",
        created_at="2026-05-30T15:00:00Z",
        started_at="2026-05-30T15:00:00Z",
        finished_at="2026-05-30T15:30:00Z",
    )

    result = merge_harness.module.validate_merge_compatibility([afternoon, noon, morning])

    assert result == (True, None, "2.5h gap detected")


def test_validation_does_not_use_created_at_for_gap_arithmetic(merge_harness):
    morning = _source(
        "morning",
        created_at="2026-05-30T10:00:00Z",
        started_at="2026-05-30T10:00:00Z",
        finished_at="2026-05-30T10:30:00Z",
    )
    startless = _source(
        "startless",
        created_at="2026-05-30T12:00:00Z",
        finished_at="2026-05-30T12:30:00Z",
    )

    result = merge_harness.module.validate_merge_compatibility([startless, morning])

    assert result == (True, None, None)
