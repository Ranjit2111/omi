from fastapi import APIRouter
from fastapi import Request, HTTPException

from models import shared
from models import task
from utils.conversations.process_conversation import process_user_expression_measurement_callback
from utils.other import hume

router = APIRouter()


@router.post('/v1/agents/hume/callback', response_model=shared.EmptyResponse, tags=['agent', 'hume', 'callback'])
def hume_expression_measurement_callback(request: Request, data: dict):
    # body untyped: external Hume AI webhook payload, forwarded wholesale to HumeJobCallbackModel.from_dict
    # which defensively parses an arbitrarily-nested prosody predictions structure. Modeling it would
    # duplicate Hume's API schema with no validation benefit since from_dict is the real parser.
    job_callback = hume.HumeJobCallbackModel.from_dict("prosody", data)
    job_id = job_callback.job_id
    # Firestore encodes request_id == None as IS_NULL, which matches any task still awaiting its Hume job id.
    if not isinstance(job_id, str) or not job_id.strip():
        raise HTTPException(status_code=400, detail="Job ID is required")

    process_user_expression_measurement_callback(task.TaskActionProvider.HUME, job_id, job_callback)

    # Empty response
    return {}
