# TODO. Model loading fails with error:
# Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
# Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
#
# But faster whisper worked in a conda environment if installed after the Hugging Face transformers.
# Try to do something with libraries.

import logging

import torch
from faster_whisper import WhisperModel

from app.transcribers.transcriber_base import SpeechTranscriber
from app.data_models import TranscriptionRequest, TranscriptionEngine, TranscriptionEngineConfig, \
    TranscriptionResponse, TranscriptSegment
from app.utils.process_utils import get_gpu_memory


logger = logging.getLogger(__name__)

class FasterWhisper(SpeechTranscriber):
    ENGINE = TranscriptionEngine.FASTER_WHISPER
    """
    ASR with a local Faster Whisper model
    """
    def __init__(self, model_name: str, config: TranscriptionEngineConfig, **model_kwargs):
        """
        model_name : str
            one of the official model names or a path to a model checkpoint.
        """
        super().__init__(model_name=model_name, config=config, **model_kwargs)

    def transcribe_file(self, request: TranscriptionRequest) -> TranscriptionResponse:
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
            torch.cuda.empty_cache()
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

    def load_model(self, model_name: str, **model_kwargs) -> None:
        if self.model_name == model_name:
            logger.debug('Model has already been loaded')
        else:
            self.unload_model()
            logger.debug(f'Starting loading model {model_name}. vRAM state:\n{get_gpu_memory()}')

            # Having to download to a specific folder because of disk space issues
            local_path = self.config.models.get(model_name)
            if local_path is None:
                error = 'Saving path for this model is not specified in the config'
                logger.error(error)
                raise ValueError(error)

            self.model = WhisperModel(
                model_name,
                download_root=local_path,
                **model_kwargs
            )
            self.model_name = model_name
            logger.debug(f'Model {model_name} loaded. vRAM state:\n{get_gpu_memory()}')
