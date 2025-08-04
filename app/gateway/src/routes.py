from typing import Annotated
import logging

from fastapi import APIRouter, Form

from app.shared.data_models import (TranscriptionRequest, TranscriptionResponse, CalculateASRMetricsRequest,
                                    ASRMetricsResponse, TranscribeCalculateASRMetricsRequest, TranscriptFormat)
from asr_manager import ASRService
from app.nlp.nlp_utils import Normalizer
from app.nlp.asr_metrics import ASRMetrics

logger = logging.getLogger(__name__)

logger.debug('Getting transcription service')
transcription_service = ASRService()

logger.debug('Getting ASR metrics service')
asr_metrics = ASRMetrics(normalizer=Normalizer())

router = APIRouter()
logger.debug('Router is up and running')

@router.post('/transcribe_file')
async def transcribe_file(request: Annotated[TranscriptionRequest, Form()]) -> None:
    """Transcribe a file loaded via a web-form"""
    request.save_to_file = True  # The only return option for web-form input is to save transcripts in files
    transcription_service.transcribe(request)

@router.post('/transcribe_api')
async def transcribe_file_via_api(request: TranscriptionRequest) -> TranscriptionResponse:
    """Transcribe a file loaded via API"""
    return transcription_service.transcribe(request)

@router.post('/eval')
async def calculate_asr_metrics(request: CalculateASRMetricsRequest) -> ASRMetricsResponse:
    """Calculate ASR quality metrics"""
    return asr_metrics(request)

@router.post('/transcribe_eval')
async def transcribe_calculate_asr_metrics(request: TranscribeCalculateASRMetricsRequest) -> ASRMetricsResponse:
    """Transcribe a file and calculate ASR quality metrics"""
    logger.debug(f'''Received request to transcribe a file and evaluate results
Request: {request}''')
    if request.transcript_formats is None:
        request.transcript_formats = [TranscriptFormat.TXT]
    if TranscriptFormat.TXT not in request.transcript_formats:
        request.transcript_formats.append(TranscriptFormat.TXT)
    result = transcription_service.transcribe(request)
    logger.debug(f'Transcript received by the router: {result}')
    if result.transcript_text is not None:
        metrics = asr_metrics(
            CalculateASRMetricsRequest(
                hypothesis=result.transcript_text,
                reference=request.reference
            )
        )
        logger.info(f'Calculated metrics: {metrics}')
        return metrics
    else:
        raise RuntimeError('Failed to calculate metrics. Transcript not available')
