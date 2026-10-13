from datetime import datetime, timezone

import pytest

import database.action_items as action_items_db
from models.action_item import ActionItemUpdateRequest
from tests.unit.fixtures.generic_firestore_fake import FakeDocumentReference, FakeFirestore

UID = 'user-reminders'
ITEM_PATH = f'users/{UID}/action_items'
UPDATED_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def store(monkeypatch):
    fake = FakeFirestore()
    monkeypatch.setattr(action_items_db, 'db', fake)
    monkeypatch.setattr(action_items_db, 'bump_action_items_list_version', lambda _uid: None)
    monkeypatch.setenv('DREAM_AGENT_MODE', 'off')
    return fake


@pytest.mark.parametrize('apple_reminder_id', ['reminder-1', None])
def test_apple_reminders_export_settles_sync_request(store, apple_reminder_id):
    store.docs[f'{ITEM_PATH}/task-1'] = {'description': 'Send budget', 'sync_requested': True}
    payload = ActionItemUpdateRequest(
        exported=True, export_platform='apple_reminders', apple_reminder_id=apple_reminder_id
    ).storage_payload()

    assert action_items_db.update_action_item(UID, 'task-1', payload) is True

    row = store.docs[f'{ITEM_PATH}/task-1']
    assert row['sync_requested'] is False
    assert row['exported'] is True
    assert row['export_platform'] == 'apple_reminders'
    assert row.get('apple_reminder_id') == apple_reminder_id
    assert action_items_db.get_pending_apple_reminders_sync(UID)['pending_export'] == []


@pytest.mark.parametrize(
    'payload',
    [
        pytest.param({'description': 'Edited task'}, id='unrelated-edit'),
        pytest.param({'exported': False}, id='export-cleared'),
        pytest.param({'exported': True, 'export_platform': 'asana'}, id='non-apple-export'),
        pytest.param({'exported': True}, id='export-without-platform'),
    ],
)
def test_update_that_is_not_an_apple_export_preserves_sync_request(store, payload):
    store.docs[f'{ITEM_PATH}/task-1'] = {'description': 'Send budget', 'sync_requested': True}

    assert action_items_db.update_action_item(UID, 'task-1', payload) is True

    assert store.docs[f'{ITEM_PATH}/task-1']['sync_requested'] is True


@pytest.mark.parametrize('pending_fields', [{}, {'exported': False}])
def test_pending_sync_drains_settled_page_so_next_poll_finds_pending_export(store, pending_fields):
    settled = {
        'description': 'Exported task',
        'sync_requested': True,
        'exported': True,
        'export_platform': 'apple_reminders',
        'updated_at': UPDATED_AT,
    }
    store.docs.update({f'{ITEM_PATH}/a{i:03}': {**settled, 'apple_reminder_id': f'reminder-{i}'} for i in range(50)})
    store.docs[f'{ITEM_PATH}/z-new'] = {'description': 'Missed push', 'sync_requested': True, **pending_fields}

    first_poll = action_items_db.get_pending_apple_reminders_sync(UID)
    second_poll = action_items_db.get_pending_apple_reminders_sync(UID)

    assert first_poll['pending_export'] == []
    assert [row['id'] for row in second_poll['pending_export']] == ['z-new']
    assert store.docs[f'{ITEM_PATH}/z-new']['sync_requested'] is True
    assert len(second_poll['synced_items']) == 50
    for i in range(50):
        assert store.docs[f'{ITEM_PATH}/a{i:03}'] == {
            **settled,
            'sync_requested': False,
            'apple_reminder_id': f'reminder-{i}',
        }


def test_pending_sync_keeps_unexported_rows_and_bidirectional_items(store):
    store.docs[f'{ITEM_PATH}/a-settled'] = {
        'description': 'Exported task',
        'sync_requested': True,
        'exported': True,
        'export_platform': 'apple_reminders',
        'apple_reminder_id': 'reminder-1',
        'updated_at': UPDATED_AT,
    }
    store.docs[f'{ITEM_PATH}/b-pending'] = {'description': 'Pending task', 'sync_requested': True}
    store.docs[f'{ITEM_PATH}/c-idle'] = {'description': 'Idle task', 'sync_requested': False}

    result = action_items_db.get_pending_apple_reminders_sync(UID)

    assert [row['id'] for row in result['pending_export']] == ['b-pending']
    assert [row['id'] for row in result['synced_items']] == ['a-settled']
    assert result['synced_items'][0]['apple_reminder_id'] == 'reminder-1'
    assert store.docs[f'{ITEM_PATH}/a-settled']['sync_requested'] is False
    assert store.docs[f'{ITEM_PATH}/a-settled']['updated_at'] == UPDATED_AT


def test_pending_sync_tolerates_settled_item_deleted_during_cleanup(store, monkeypatch):
    store.docs[f'{ITEM_PATH}/a-settled'] = {'sync_requested': True, 'exported': True}
    store.docs[f'{ITEM_PATH}/b-pending'] = {'description': 'Pending task', 'sync_requested': True}
    update = FakeDocumentReference.update

    def delete_then_update(ref, data):
        store.docs.pop(ref.path)
        update(ref, data)

    monkeypatch.setattr(FakeDocumentReference, 'update', delete_then_update)

    result = action_items_db.get_pending_apple_reminders_sync(UID)

    assert [row['id'] for row in result['pending_export']] == ['b-pending']
    assert f'{ITEM_PATH}/a-settled' not in store.docs
