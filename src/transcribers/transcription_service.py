import gc
import logging
from importlib import import_module

import torch

from src.data_models import TranscriptionEngine, TranscriptionRequest, TranscriptionResponse, TRANSCRIPTION_ENGINE_CONFIGS
from src.utils.process_utils import get_gpu_memory


logger = logging.getLogger(__name__)

class TranscriptionService:
    """Class to manage transcriber loading, unloading and use"""
    def __init__(self):
        self.transcriber = None
        logger.debug('Transcription service initialization')

    def unload_transcriber(self) -> None:
        """Unload the current transcriber model and clean up memory"""
        logger.debug(f'Starting to unload transcriber. Memory state:\n {get_gpu_memory()}')
        if self.transcriber is not None:
            del self.transcriber
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.transcriber = None
        logger.debug(f'Transcriber unloaded. Memory state:\n {get_gpu_memory()}')

    def load_transcriber(self, engine: TranscriptionEngine, model_name: str) -> None:
        """Load a transcription model with the specified model"""
        logger.debug(f'Starting loading engine {engine} with model {model_name}')
        if self.transcriber:
            self.unload_transcriber()
        engine_config = TRANSCRIPTION_ENGINE_CONFIGS.get(engine)
        if not engine_config:
            logger.error(f'No config for engine: {engine}')
            raise ValueError(f'No config for engine: {engine}')
        module = import_module(f'src.transcribers.{engine.value}')
        class_ = getattr(module, engine_config.transcriber_class)
        self.transcriber = class_(model_name=model_name, config=engine_config)
        logger.debug(f'Transcriber loaded: {self.transcriber}.\nMemory state:\n {get_gpu_memory()}')

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        logger.info(f'Received a transcription request: {request}')

        # Check that the model name can be used with the engine
        if request.model_name is None:
            model_name = TRANSCRIPTION_ENGINE_CONFIGS[request.engine].default_model
        elif request.model_name not in TRANSCRIPTION_ENGINE_CONFIGS[request.engine].models.keys():
            error = f'{request.engine} cannot use model {request.model_name}'
            logger.error(error)
            raise ValueError(error)
        else:
            model_name = request.model_name

        if self.transcriber is None or self.transcriber.ENGINE != request.engine:
            logger.debug('Loading a new transcriber')
            self.load_transcriber(engine=request.engine, model_name=model_name)
        if self.transcriber.model_name != model_name:
            logger.debug(f'''Loading a new model for the existing transcriber. 
            Transcriber model: {self.transcriber.model_name}.
            Requested model: {model_name}
''')
            self.transcriber.load_model(model_name=model_name)
        logger.debug(f'Ready to start transcription. Memory state:\n {get_gpu_memory()}')
        result = self.transcriber(request=request)
        logger.debug(f'Transcription finished. Memory state:\n {get_gpu_memory()}')
        return result


