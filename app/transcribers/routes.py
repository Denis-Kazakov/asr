import logging
import os
from importlib import import_module

from fastapi import APIRouter

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionResponse

t = import_module(os.getenv('TRANSCRIBER_MODULE'))

logger = logging.getLogger(__name__)

logger.debug('Getting transcription service')
transcriber = t.SpeechTranscriber()

logger.debug('Router is up and running')
router = APIRouter()


@router.post('/transcription_service/transcribe')
async def transcribe_file(request: TranscriptionServiceRequest) -> TranscriptionResponse:
    """Transcribe a file"""
    return await transcriber(request)

