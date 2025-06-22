import gc
import json
import logging
import os
from datetime import datetime
from typing import Literal

import torch.cuda

from src.data_models import TranscriptionRequest, TranscriptionEngineConfig
from src.utils.process_utils import get_gpu_memory

logger = logging.getLogger(__name__)

class SpeechTranscriber:
    """Base class for all speech transcribers"""
    OUTPUT_PATH = 'output'
    ENGINE = None  # Must be defined in child classes
    def __init__(self, model_name: str, config: TranscriptionEngineConfig):
        os.makedirs(self.__class__.OUTPUT_PATH, exist_ok = True)
        self.model = None
        self.model_name = None
        self.config = config
        if self.__class__.ENGINE is None:
            raise ValueError('A child class of SpeechTranscriber must define the ENGINE class variable')
        try:
            self.load_model(model_name=model_name)
            logger.debug('ASR model loaded')
        except Exception as e:
            logger.exception(f'Failed to load the model with error: {str(e)}')


    def __call__(self, request: TranscriptionRequest) -> None:
        """
        Transcribe an audio file
        """
        raise NotImplementedError

    def load_model(self, model_name: str) -> None:
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

    def _save_transcript(self, transcript: str | dict, mode: Literal['text', 'json']) -> None:
        t = datetime.now()
        if mode == 'text':
            saving_path = os.path.join(self.__class__.OUTPUT_PATH, f'transcript_{t}.txt')
            with open(saving_path, 'w') as f:
                f.write(transcript)
            logger.debug(f'Transcript text saved at {saving_path}')

        elif mode == 'json':
            saving_path = os.path.join(self.__class__.OUTPUT_PATH, f'transcript_{t}.json')
            with open(saving_path, 'w') as f:
                json.dump(transcript, f)
            logger.debug(f'Transcript segments saved at {saving_path}')


