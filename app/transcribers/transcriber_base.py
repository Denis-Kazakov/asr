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
        if self.model_spec is not None and self.model_spec != request.model_spec:
            await self.unload_model()
        if self.model_spec is None:
            await self.load_model(model_spec=request.model_spec)
        return await self.transcribe(request)

    async def transcribe(self, request: TranscriptionServiceRequest) -> TranscriptionResponse:
        raise NotImplementedError

    async def load_model(self, model_spec: ModelSpec) -> None:
        """
        Load an ASR model
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


