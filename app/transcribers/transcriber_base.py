import gc
import logging

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionResponse, ModelSpec

logger = logging.getLogger(__name__)

class SpeechTranscriberBase:
    """Base class for all speech transcribers"""
    def __init__(self):
        self.model_spec = None
        self.model = None

    async def __call__(self, request: TranscriptionServiceRequest) -> TranscriptionResponse:
        """Transcribe an audio file"""
        logger.debug(f'Transcription service received request for transcription: {request}')
        if request.model_spec != self.model_spec:
            self.model, self.model_spec = await self.load_model(model_spec=request.model_spec)
        return await self.transcribe(request)

    async def transcribe(self, request: TranscriptionServiceRequest) -> TranscriptionResponse:
        raise NotImplementedError

    async def load_model(self, model_spec: ModelSpec) -> tuple:
        """
        Load an ASR model
        Before loading, all subclasses should check if another model has been loaded and unload it first.
        Output:
            A tuple of:
                - Instance of an ASR model
                - model_spec
        """
        raise NotImplementedError

    async def unload_model(self) -> None:
        """Unload the current transcription model and clean up memory"""
        if self.model is not None:
            logger.debug(f'Starting model unloading')
            del self.model
            gc.collect()
            self.model = None
            self.model_spec = None
            logger.debug(f'Model unloaded')
        else:
            logger.debug('No need to unload. Model is None')


