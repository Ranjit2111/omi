"""POST /v1/agents/hume/callback rejects a callback with no usable job_id and reports processing faults as a sanitized 500."""

import importlib
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.task import TaskActionProvider
from tests.unit.memory_import_isolation import restore_sys_modules, snapshot_sys_modules

_ISOLATED_MODULES = ('utils.conversations.process_conversation', 'routers.agents')


@pytest.fixture
def process_callback():
    saved = snapshot_sys_modules(_ISOLATED_MODULES)
    stub = MagicMock()
    process_module = types.ModuleType('utils.conversations.process_conversation')
    process_module.process_user_expression_measurement_callback = stub
    sys.modules['utils.conversations.process_conversation'] = process_module
    sys.modules.pop('routers.agents', None)
    yield stub
    restore_sys_modules(saved)


@pytest.fixture
def client(process_callback):
    app = FastAPI()
    app.include_router(importlib.import_module('routers.agents').router)
    return TestClient(app)


def test_valid_callback_is_processed_and_acknowledged(client, process_callback):
    response = client.post('/v1/agents/hume/callback', json={'job_id': 'job-1', 'status': 'COMPLETED'})

    assert response.status_code == 200
    assert response.json() == {}
    provider, request_id, callback = process_callback.call_args.args
    assert (provider, request_id) == (TaskActionProvider.HUME, 'job-1')
    assert callback.status == 'COMPLETED'


@pytest.mark.parametrize(
    'payload',
    [
        pytest.param({}, id='empty-body'),
        pytest.param({'status': 'COMPLETED'}, id='missing-job-id'),
        pytest.param({'job_id': None, 'status': 'COMPLETED'}, id='null-job-id'),
        pytest.param({'job_id': '', 'status': 'COMPLETED'}, id='empty-job-id'),
        pytest.param({'job_id': '   ', 'status': 'COMPLETED'}, id='blank-job-id'),
        pytest.param({'job_id': 123, 'status': 'COMPLETED'}, id='non-string-job-id'),
    ],
)
def test_callback_without_usable_job_id_is_rejected_before_any_lookup(client, process_callback, payload):
    response = client.post('/v1/agents/hume/callback', json=payload)

    assert response.status_code == 400
    assert response.json() == {'detail': 'Job ID is required'}
    process_callback.assert_not_called()


def test_processing_failure_returns_sanitized_500(client, process_callback, caplog):
    process_callback.side_effect = RuntimeError('deadline exceeded for token AbCd1234EfGh5678')

    response = client.post('/v1/agents/hume/callback', json={'job_id': 'job-1', 'status': 'COMPLETED'})

    assert response.status_code == 500
    assert response.json() == {'detail': 'Failed to process Hume callback'}
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert (
        errors[0].getMessage()
        == 'Hume callback processing failed: RuntimeError: deadline exceeded for token AbCd***5678'
    )
    assert 'AbCd1234EfGh5678' not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
