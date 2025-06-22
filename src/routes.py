from typing import Annotated
import logging

from fastapi import APIRouter, Form

from src.data_models import TranscriptionRequest
from src.transcribers.asr_manager import TranscriptionService


logger = logging.getLogger(__name__)

logger.debug('Getting transcription service')
transcription_service = TranscriptionService()
logger.debug('Router is up and running')
router = APIRouter()

@router.post('/transcribe_file')
def transcribe_file(request: Annotated[TranscriptionRequest, Form()]):
    transcription_service.transcribe(request)