import logging

import torch
import whisper
from accelerate import Accelerator

from app.transcribers.transcriber_base import SpeechTranscriber
from app.data_models import TranscriptionRequest, TranscriptionEngine, TranscriptionEngineConfig, \
    TranscriptionResponse
from app.utils.process_utils import get_gpu_memory


logger = logging.getLogger(__name__)

class LocalWhisper(SpeechTranscriber):
    ENGINE = TranscriptionEngine.WHISPER_LOCAL
    """
    ASR with a local Whisper model
    """
    def __init__(self, model_name: str, config: TranscriptionEngineConfig, **model_kwargs):
        """
        model_name : str
            one of the official model names listed by `whisper.available_models()`, or
            path to a model checkpoint containing the model dimensions and the model state_dict.
        """
        super().__init__(model_name=model_name, config=config, **model_kwargs)

    def transcribe_file(self, request: TranscriptionRequest) -> TranscriptionResponse:
        try:
            with torch.no_grad():
                result = self.model.transcribe(
                    audio=str(request.filepath),
                    language=request.language_code,
                    **request.transcription_kwargs
                )
            torch.cuda.empty_cache()
            logger.info(f'Transcript ready. Text: {result['text'][:100]}...')
            return TranscriptionResponse(
                transcript_text=result['text'],
                transcript_segments=result['segments'],
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
            accelerator = Accelerator()

            # Having to download to a specific folder because of disk space issues
            local_path = self.config.models.get(model_name)
            if local_path is None:
                error = 'Saving path for this model is not specified in the config'
                logger.error(error)
                raise ValueError(error)

            model_kwargs['device'] = 'cpu'
            self.model = whisper.load_model(
                name=local_path,
                **model_kwargs
            )\
                .to(accelerator.device)
            self.model_name = model_name
            logger.debug(f'Model {model_name} loaded. vRAM state:\n{get_gpu_memory()}')
