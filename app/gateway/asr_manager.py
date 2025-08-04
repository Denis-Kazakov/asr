import os
import json
import logging
from dotenv import load_dotenv

import docker
from pydantic import FilePath
import httpx

from app.shared.data_models import TranscriptionServiceRequest, TranscriptionRequest, TranscriptionResponse, \
    TRANSCRIPTION_ENGINE_CONFIGS, TranscriptFormat, ModelSpec
from app.nlp.nlp_utils import segments2srt

logger = logging.getLogger(__name__)
load_dotenv('../.env')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

class ASRService:
    """
    Class to handle ASR requests as well as to manage transcriber (Docker container) loading, unloading and use.
    Due to limited vRAM, only a single container using GPU is allowed to run at a time
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.gpu_container = None  #Container requiring GPU (only one is allowed to run at a time)
        logger.debug('Transcription service initialized')

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        logger.info(f'Received a transcription request: {request}')

        # If a wrong container is running, stop it and start the right one
        requested_container_name = TRANSCRIPTION_ENGINE_CONFIGS[request.engine].container_name
        requested_image = TRANSCRIPTION_ENGINE_CONFIGS[request.engine].docker_image
        if self.gpu_container is not None and self.gpu_container.attrs['name'] != requested_container_name:
            self.gpu_container.stop()
            self.gpu_container = None
        if self.gpu_container is None:
            self.gpu_container = self.docker_client.containers.run(
                image=requested_image,
                detach=True,
                name=requested_container_name,
                remove=True,
                ports={'3001/tcp': 3001},
                volumes={
                    TRANSCRIPTION_ENGINE_CONFIGS[request.engine].base_path: {'bind': '/app/asr_models'},
                    'mode': 'ro'
                }
            )

        # Check that the model name can be used with the engine
        # TODO. Implement checking other model specs
        if request.model_spec is None or request.model_spec.model_name is None:
            request.model_spec = ModelSpec(model_name=TRANSCRIPTION_ENGINE_CONFIGS[request.engine].default_model)
        if request.model_spec.model_name not in TRANSCRIPTION_ENGINE_CONFIGS[request.engine].models.keys():
            error = f'{request.engine} cannot use model {request.model_spec.model_name}'
            logger.error(error)
            raise ValueError(error)

        # Transcription
        logger.info(f'Ready to start transcription')
        try:
            response = httpx.post(
                f'http://127.0.0.1:3001/transcription_service/transcribe',
                json=TranscriptionServiceRequest(**request.model_dump())
            )
            response.raise_for_status()
            logger.info(f'Transcription finished')
            response = TranscriptionResponse(**response.json())
            if request.save_to_file:
                await self._save_transcript(
                    transcription_output=response,
                    source_path=request.filepath,
                    transcript_formats=request.transcript_formats
                )
            return response
        except httpx.RequestError as exc:
            message = f'Transcription request failed with error {str(exc)}'
            logger.error(message)
            return TranscriptionResponse(
                error=message
            )
        except httpx.HTTPStatusError as exc:
            message = f"Transcription request received an error response {str(exc)}"
            logger.error(message)
            return TranscriptionResponse(
                error=message
            )
        except Exception as exc:
            message = f'Transcription failed with error{str(exc)}'
            logger.error(message)
            return TranscriptionResponse(
                error=message
            )


    @staticmethod
    async def _save_transcript(
            transcription_output: TranscriptionResponse,
            source_path: FilePath,
            transcript_formats: list | None
    ) -> None:
        """
        Save the transcript in all specified formats.
        If no formats are specified, save in all formats.
        """
        filename_stem = os.path.splitext(os.path.basename(source_path))[0]
        if transcript_formats is None or TranscriptFormat.TXT in transcript_formats:
            saving_path = os.path.join(OUTPUT_PATH, f'{filename_stem}.txt')
            with open(saving_path, 'w') as f:
                f.write(transcription_output.transcript_text)
            logger.debug(f'Transcript text saved at {saving_path}')
        if transcript_formats is None or TranscriptFormat.SRT in transcript_formats:
            saving_path = os.path.join(OUTPUT_PATH, f'{filename_stem}.srt')
            with open(saving_path, 'w') as f:
                f.write(segments2srt(transcription_output.transcript_segments))
            logger.debug(f'Transcript in the SRT format saved at {saving_path}')
        if transcript_formats is None or TranscriptFormat.JSON in transcript_formats:
            saving_path = os.path.join(OUTPUT_PATH, f'{filename_stem}.json')
            with open(saving_path, 'w') as f:
                output = [segment.model_dump() for segment in transcription_output.transcript_segments]
                json.dump(output, f)
            logger.debug(f'Transcript JSON saved at {saving_path}')

