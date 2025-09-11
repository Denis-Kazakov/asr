import logging
import os
from importlib import import_module

from fastapi import APIRouter

from app.shared.data_models import SegmentationRequest, TranscriptionResponse, ServiceState

logger = logging.getLogger(__name__)

# Import sentence tokenization module
try:
    s = import_module(os.getenv('SENTENCE_TOKENIZER_MODULE'))
    logger.debug('Sentence tokenization module loaded')
except Exception as e:
    logger.exception(f'Failed to import sentence tokenization module with error: {str(e)}', exc_info=True)
try:
    module = os.getenv('SENTENCE_TOKENIZER_MODULE')
    logger.debug(f'Module to be loaded: {module}')
    s = import_module(module)
    logger.debug('Sentence tokenizer module loaded')
    segmenter = s.Segmenter()
except Exception as e:
    message = f'Failed to import sentence tokenization module with error: {str(e)}'
    logger.exception(message, exc_info=True)
    raise RuntimeError(message)


router = APIRouter()

@router.get('/nlp/segmentation_service_state')
async def check_service_state() -> ServiceState:
    """Return the current state of the service: is it ready to transcribe?"""
    if not segmenter:
        return ServiceState(
            healthy=False,
            details='Segmenter not loaded'
        )
    else:
        return ServiceState(
            healthy=True,
            details='Segmenter ready to receive requests'
        )
@router.post('/nlp/segmentation')
async def resegment_transcript(request: SegmentationRequest) -> TranscriptionResponse:
    """Transcribe a file"""
    return await segmenter(request)
