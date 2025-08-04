import logging
import gc

import torch
import whisper
from accelerate import Accelerator

from app.transcribers.transcriber_base import SpeechTranscriberBase
from app.shared.data_models import TranscriptionServiceRequest, ModelSpec, TranscriptionResponse
from app.shared.process_utils import get_gpu_memory


logger = logging.getLogger(__name__)

class SpeechTranscriber(SpeechTranscriberBase):
    """
    ASR with a local Whisper model
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.accelerator = Accelerator()

    async def transcribe(self, request: TranscriptionServiceRequest) -> TranscriptionResponse:
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

    async def load_model(self, model_spec: ModelSpec) -> None:
        if self.model_spec == model_spec:
            logger.debug('Model has already been loaded')
        else:
            await self.unload_model()
            logger.debug(f'Starting loading model {model_spec.model_name}. vRAM state:\n{get_gpu_memory()}')
            self.model = whisper.load_model(
                name=model_spec.model_name,
                device='cpu',  # Forced to CPU to use the accelerator
                download_root=model_spec.model_path,
                **model_spec.model_kwargs
            )\
                .to(self.accelerator.device)
            self.model_spec = model_spec
            logger.debug(f'Model {model_spec.model_name} loaded. vRAM state:\n{get_gpu_memory()}')

    async def unload_model(self) -> None:
        """
        Unload the current transcription model and clean up memory
        Overriding the base class method to use torch
        """
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
