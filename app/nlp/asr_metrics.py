import logging

from jiwer import wer, cer

from app.shared.data_models import CalculateASRMetricsRequest, ASRMetricsResponse

logger = logging.getLogger(__name__)


class ASRMetrics:
    """Calculate ASR quality metrics"""
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, request: CalculateASRMetricsRequest) -> ASRMetricsResponse:
        return ASRMetricsResponse(
            wer=wer(reference=request.reference, hypothesis=request.hypothesis),
            cer=cer(reference=request.reference, hypothesis=request.hypothesis),
            wer_normalized=wer(
                reference=self.normalizer.inverse_normalize(request.reference),
                hypothesis=self.normalizer.inverse_normalize(request.hypothesis)
            ),
            cer_normalized=cer(
                reference=self.normalizer.inverse_normalize(request.reference),
                hypothesis=self.normalizer.inverse_normalize(request.hypothesis)
            )
        )

