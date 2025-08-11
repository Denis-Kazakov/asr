import logging

from faster_whisper import WhisperModel

from app.transcribers.transcriber_base import SpeechTranscriberBase
from app.shared.data_models import TranscriptionServiceRequest, ModelSpec, TranscriptionResponse, TranscriptSegment


logger = logging.getLogger(__name__)

class SpeechTranscriber(SpeechTranscriberBase):
    """
    ASR with a local Faster Whisper model
    """
    def __init__(self):
        """
        """
        super().__init__()

    async def transcribe(self, request: TranscriptionServiceRequest) -> TranscriptionResponse:
        try:
            segments, info = self.model.transcribe(
                str(request.filepath),
                language=request.language_code,
                **request.transcription_kwargs
            )
            transcript_text = []
            transcript_segments = []
            for segment in segments:
                transcript_text.append(segment.text)
                transcript_segments.append(TranscriptSegment(start=segment.start, end=segment.end, text=segment.text))
            logger.info(f'Transcript ready. Text: {transcript_text[:5]}...')
            return TranscriptionResponse(
                transcript_text=' '.join(transcript_text),
                transcript_segments=transcript_segments,
                error=None
            )
        except Exception as e:
            logger.exception(f'Failed to transcribe file with error: {str(e)}')
            return TranscriptionResponse(
                transcript_text=None,
                transcript_segments=None,
                error=str(e)
            )

    async def load_model(self, model_spec: ModelSpec) -> tuple:
        """
        Load an ASR model
        Output:
            A tuple of:
                - Instance of an ASR model
                - model_spec
        """
        if self.model is not None:
            await self.unload_model()
        logger.debug(f'Starting loading model {model_spec.model_name}')

        model = WhisperModel(
            model_size_or_path=model_spec.model_name,
            download_root='/project/app/transcribers/asr_models',
            device=model_spec.device,
            **model_spec.model_kwargs
        )
        logger.debug(f'Model {model_spec.model_name} loaded')
        return model, model_spec
