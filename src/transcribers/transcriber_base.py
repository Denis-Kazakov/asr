import gc
import json
import logging
import os
from typing import Literal

import torch.cuda

from src.data_models import TranscriptionRequest, TranscriptionResponse, TranscriptionEngineConfig, TranscriptFormat
from src.utils.process_utils import get_gpu_memory

logger = logging.getLogger(__name__)

class SpeechTranscriber:
    """Base class for all speech transcribers"""
    OUTPUT_PATH = 'output'
    ENGINE = None  # Must be defined in child classes

    def __init__(self, model_name: str, config: TranscriptionEngineConfig, **model_kwargs):
        os.makedirs(self.__class__.OUTPUT_PATH, exist_ok = True)
        self.model = None
        self.model_name = None
        self.config = config
        if self.__class__.ENGINE is None:
            raise ValueError('A child class of SpeechTranscriber must define the ENGINE class variable')
        try:
            self.load_model(model_name=model_name, **model_kwargs)
            logger.debug('ASR model loaded')
        except Exception as e:
            logger.exception(f'Failed to load the model with error: {str(e)}')

    def __call__(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Get an audio file transcribed and handle other tasks such as saving the results
        """
        result = self.transcribe_file(request=request)
        if result.error is None:
            if request.save_to_file:
                try:
                    filename_stem = os.path.splitext(os.path.basename(request.filepath))[0]
                    if request.transcript_formats is None or TranscriptFormat.TXT in request.transcript_formats:
                        self._save_transcript(transcript=result.transcript_text, filename_stem=filename_stem, mode='text')
                    elif TranscriptFormat.SRT in request.transcript_formats:
                        # TODO. This is wrong: SRT instead of JSON
                        self._save_transcript(transcript=result.transcript_segments, filename_stem=filename_stem, mode='json')
                    else:
                        result.error = 'Saving failed.No acceptable format for saving the transcript'
                        logger.error(result.error)
                except Exception as e:
                    result.error = f'Saving failed with error: {str(e)}'
                    logger.exception(result.error)
        return result

    def transcribe_file(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe an audio file"""
        raise NotImplementedError

    def load_model(self, model_name: str, **model_kwargs) -> None:
        """
        Load an ASR model
        """
        raise NotImplementedError

    def unload_model(self) -> None:
        """Unload the current transcription model and clean up memory"""
        if self.model is not None:
            logger.debug(f'Starting model unloading. vRAM state:\n{get_gpu_memory()}')
            if isinstance(self.model, torch.nn.Module):
                self.model.to('cpu')
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            logger.debug(f'Model unloaded. vRAM state:\n{get_gpu_memory()}')

        else:
            logger.debug('No need to unload. Model is None')

    def _save_transcript(self, transcript: str | dict, filename_stem: str, mode: Literal['text', 'json']) -> None:
        if mode == 'text':
            saving_path = os.path.join(self.__class__.OUTPUT_PATH, f'{filename_stem}.txt')
            with open(saving_path, 'w') as f:
                f.write(transcript)
            logger.debug(f'Transcript text saved at {saving_path}')

        elif mode == 'json':
            saving_path = os.path.join(self.__class__.OUTPUT_PATH, f'{filename_stem}.json')
            with open(saving_path, 'w') as f:
                json.dump(transcript, f)
            logger.debug(f'Transcript segments saved at {saving_path}')


