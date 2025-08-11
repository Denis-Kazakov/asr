import logging
import os
from importlib import import_module

from fastapi import APIRouter

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionResponse, TranscriptionServiceState

t = import_module(os.getenv('TRANSCRIBER_MODULE'))

logger = logging.getLogger(__name__)

logger.debug('Getting transcription service')
transcriber = t.SpeechTranscriber()

logger.debug('Router is up and running')
router = APIRouter()


@router.get('/transcription_service/state')
async def check_service_state() -> TranscriptionServiceState:
    """Return the current state of the service: is it ready to transcribe?"""
    if not transcriber:
        return TranscriptionServiceState(
            healthy=False,
            details='Transcriber not loaded'
        )
    else:
        return TranscriptionServiceState(
            healthy=True,
            details=f'Transcriber ready to receive requests. Model spec: {transcriber.model_spec}'
        )

@router.post('/transcription_service/transcribe')
async def transcribe_file(request: TranscriptionServiceRequest) -> TranscriptionResponse:
    """Transcribe a file"""
    return await transcriber(request)
