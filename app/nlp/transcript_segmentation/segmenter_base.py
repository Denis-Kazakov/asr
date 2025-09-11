import logging

from app.shared.data_models import SegmentationRequest, TranscriptionResponse

logger = logging.getLogger(__name__)

class SegmenterBase:
    """
    Base class for changing a transcript segmentation, e.g. from random segments into complete sentences,
    making sure each has correct start and end timestamps.
    """
    def __init__(self):
        pass

    def __call__(self, request: SegmentationRequest) -> TranscriptionResponse:
        raise NotImplementedError




