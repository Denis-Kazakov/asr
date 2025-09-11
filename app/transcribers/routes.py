import logging
import os
from importlib import import_module

from fastapi import APIRouter, Depends

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionResponse, ServiceState

logger = logging.getLogger(__name__)

try:
    module = os.getenv('TRANSCRIBER_MODULE')
    logger.debug(f'Module to be loaded: {module}')
    t = import_module(module)
    logger.debug('Transcriber module loaded')
    transcriber = t.SpeechTranscriber()
except Exception as e:
    message = f'Failed to import transcriber module with error: {str(e)}'
    logger.exception(message, exc_info=True)
    raise RuntimeError(message)


router = APIRouter()

@router.get('/transcription_service/state')
async def check_service_state() -> ServiceState:
    """Return the current state of the service: is it ready to transcribe?"""
    if not transcriber:
        return ServiceState(
            healthy=False,
            details='Transcriber not loaded'
        )
    else:
        return ServiceState(
            healthy=True,
            details=f'Transcriber ready to receive requests. Model spec: {transcriber.model_spec}'
        )

@router.post('/transcription_service/transcribe')
async def transcribe_file(request: TranscriptionServiceRequest) -> TranscriptionResponse:
    """Transcribe a file"""
    return await transcriber(request)
