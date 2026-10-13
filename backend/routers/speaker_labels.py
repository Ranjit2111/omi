import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import ValidationError

from database import conversations as conversations_db
from database import voice_profiles as voice_profiles_db
from models.conversation import Conversation
from models.speaker_labels import RejectSpeakerRequest, VoiceMatchesResponse
from utils.conversations.factory import deserialize_conversation
from utils.log_sanitizer import sanitize, sanitize_validation_error
from utils.other import endpoints as auth
from utils.other.storage import delete_speech_profile_blob
from utils.speaker_voice_matches import find_person_voice_matches

logger = logging.getLogger(__name__)

router = APIRouter()

_REJECT_FAILED_DETAIL = 'Unable to process speaker rejection'


def _log_text(error: Exception) -> str:
    return sanitize_validation_error(error) if isinstance(error, ValidationError) else sanitize(error)


def _internal_error(operation: str, uid: str, error: Exception, detail: str) -> HTTPException:
    logger.error(f'{operation} failed uid={uid}: {type(error).__name__}: {_log_text(error)}')
    return HTTPException(status_code=500, detail=detail)


@router.post(
    '/v1/conversations/{conversation_id}/speakers/{speaker_id}/reject',
    response_model=Conversation,
    tags=['conversations'],
)
def reject_speaker_label(
    conversation_id: str,
    speaker_id: int,
    data: RejectSpeakerRequest,
    background_tasks: BackgroundTasks,
    uid: str = Depends(auth.get_current_user_uid),
):
    if not conversation_id.strip():
        raise HTTPException(status_code=400, detail='conversation_id must not be blank')
    if speaker_id < 0:
        raise HTTPException(status_code=400, detail='speaker_id must not be negative')
    try:
        raw, resolved, removed, _before = conversations_db.assign_conversation_speaker(
            uid,
            conversation_id,
            person_id=None,
            is_user=False,
            speaker_id=speaker_id,
            segment_ids=data.segment_ids,
            use_for_speech_training=False,
            rejection={'kind': data.kind, 'person_id': data.person_id},
        )
    except LookupError as error:
        logger.warning(f'reject_speaker_label target not found uid={uid}: {_log_text(error)}')
        raise HTTPException(status_code=404, detail='Conversation, speaker or person not found') from error
    except PermissionError as error:
        raise HTTPException(status_code=402, detail='A paid plan is required to access this conversation.') from error
    except ValueError as error:
        logger.warning(f'reject_speaker_label request rejected uid={uid}: {_log_text(error)}')
        raise HTTPException(status_code=409, detail='Invalid speaker rejection request') from error
    except Exception as error:
        raise _internal_error('reject_speaker_label', uid, error, _REJECT_FAILED_DETAIL) from error
    try:
        if data.kind == 'not_a_person':
            resolved_ids = set(resolved)
            affected = {
                int(segment['speaker_id'])
                for segment in raw.get('transcript_segments') or []
                if segment.get('id') in resolved_ids and isinstance(segment.get('speaker_id'), int)
            } or {speaker_id}
            generation = (raw.get('manual_speaker_assignments') or {}).get('generation')
            for affected_speaker in sorted(affected):
                voice_profiles_db.record_ignored_voice(
                    uid, raw['id'], affected_speaker, datetime.now(timezone.utc), assignment_generation=generation
                )
        for path in removed:
            background_tasks.add_task(delete_speech_profile_blob, path)
        return deserialize_conversation(raw)
    except Exception as error:
        raise _internal_error('reject_speaker_label post-processing', uid, error, _REJECT_FAILED_DETAIL) from error


@router.get('/v1/users/people/{person_id}/voice-matches', response_model=VoiceMatchesResponse, tags=['v1'])
async def get_person_voice_matches(person_id: str, uid: str = Depends(auth.get_current_user_uid)):
    if not person_id.strip():
        raise HTTPException(status_code=400, detail='person_id must not be blank')
    try:
        return await find_person_voice_matches(uid, person_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail='Person not found') from error
    except Exception as error:
        raise _internal_error('get_person_voice_matches', uid, error, 'Unable to load voice matches') from error
