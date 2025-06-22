import logging

import torch
import whisper
from accelerate import Accelerator

from src.transcribers.transcriber_base import SpeechTranscriber
from src.data_models import TranscriptionRequest, TranscriptFormat, TranscriptionEngine, TranscriptionEngineConfig
from src.utils.process_utils import get_gpu_memory


logger = logging.getLogger(__name__)

class LocalWhisper(SpeechTranscriber):
    ENGINE = TranscriptionEngine.WHISPER_LOCAL
    """
    ASR with a local Whisper model
    """
    def __init__(self, model_name: str, config: TranscriptionEngineConfig):
        """
        model_name : str
            one of the official model names listed by `whisper.available_models()`, or
            path to a model checkpoint containing the model dimensions and the model state_dict.
        """
        super().__init__(model_name=model_name, config=config)

    def __call__(self, request: TranscriptionRequest) -> None:
        try:
            with torch.no_grad():
                result = self.model.transcribe(
                    audio=str(request.filepath),
                    fp16=(request.compute_type is None or request.compute_type=='fp16'),
                    language=request.language_code,
                )
            torch.cuda.empty_cache()
            logger.info(f'Transcript ready. Text: {result['text'][:100]}...')
            if request.transcript_formats is None or TranscriptFormat.TXT in request.transcript_formats:
                self._save_transcript(transcript= result['text'], mode='text')
            elif TranscriptFormat.SRT in request.transcript_formats:
                self._save_transcript(transcript=result['segments'], mode='json')
            else:
                raise ValueError('No acceptable format for saving the transcript')
        except Exception as e:
            logger.exception(f'Failed to transcribe file with error: {str(e)}')

    def load_model(self, model_name: str) -> None:
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
                logger.error('Saving path for this model is not specified in the config')
                raise ValueError(error)

            self.model = whisper.load_model(
                name=local_path,
                device='cpu',
            )\
                .to(accelerator.device)
            self.model_name = model_name
            logger.debug(f'Model {model_name} loaded. vRAM state:\n{get_gpu_memory()}')
