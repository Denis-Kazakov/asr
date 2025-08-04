import logging
import os
from importlib import import_module
from dotenv import load_dotenv

from fastapi import APIRouter

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionResponse

load_dotenv('../.env')
TRANSCRIBER_PATH = os.getenv('TRANSCRIBER_PATH')
t = import_module(TRANSCRIBER_PATH)

logger = logging.getLogger(__name__)

logger.debug('Getting transcription service')
transcriber = t.SpeechTranscriber()

logger.debug('Router is up and running')
router = APIRouter()


@router.post('/transcription_service/transcribe')
async def transcribe_file(request: TranscriptionServiceRequest) -> TranscriptionResponse:
    """Transcribe a file"""
    return await transcriber(request)

