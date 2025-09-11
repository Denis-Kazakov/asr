import logging

from faster_whisper import WhisperModel

from app.transcribers.transcriber_base import SpeechTranscriberBase
from app.shared.data_models import TranscriptionServiceRequest, ModelSpec, TranscriptionResponse, TranscriptSegment,TimeStampedWord


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
                word_timestamps=request.word_timestamps,
                **request.transcription_kwargs
            )
            logger.debug(f'Transcript metadata: {info}')
            transcript_text = []
            transcript_segments = []
            for segment in segments:
                logger.debug(f'Segment: {segment}')
                transcript_text.append(segment.text)
                if request.word_timestamps:
                    words = [TimeStampedWord(start=word.start, end=word.end, word=word.word) for word in segment.words]
                else:
                    words = None
                transcript_segments.append(
                    TranscriptSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words
                    )
                )

            logger.info(f'Transcript ready. Text: {transcript_text[:5]}...')
            return TranscriptionResponse(
                transcript_text=' '.join(transcript_text),
                transcript_segments=transcript_segments,
                language_code=info.language,
                error=None
            )
        except Exception as e:
            logger.exception(f'Failed to transcribe file with error: {str(e)}')
            return TranscriptionResponse(
                transcript_text=None,
                transcript_segments=None,
                language_code=request.language_code,
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
