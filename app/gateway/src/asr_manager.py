import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

import docker
import httpx
import librosa
from pydantic import FilePath

from app.nlp.nlp_utils import segments2srt
from app.shared.data_models import TranscriptionServiceRequest, TranscriptionRequest, TranscriptionResponse, \
    TranscriptFormat, ModelSpec, ServiceState, SegmentationRequest, SegmentationType
from app.shared.model_configs import TRANSCRIPTION_ENGINE_CONFIGS, SENTENCE_TOKENIZER_CONFIGS

logger = logging.getLogger(__name__)

class ASRService:
    """
    Class to handle ASR requests as well as to manage transcriber (Docker container) loading, unloading and use.
    Due to limited vRAM, only a single container using GPU is allowed to run at a time
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.transcriber_container = None  # Container requiring GPU (only one is allowed to run at a time)
        self.sentence_tokenizer_container = None  # Will run on CPU (but also only one at a time)
        with open('sentence_tokenizers.json') as f:
            self.sentence_tokenizer_selector = json.load(f)

        # Input and output dirs
        script_dir = Path(__file__).parent.resolve()
        top_level_dir = script_dir.parent.parent
        self.input_dir = os.path.join(top_level_dir, 'input')
        self.output_dir = os.path.join(top_level_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f'Input dir: {self.input_dir}')
        logger.info('Transcription service initialized')


    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        logger.info(f'Received a transcription request: {request}')

        # If a wrong container is running, stop it and start the right one
        requested_container_name = TRANSCRIPTION_ENGINE_CONFIGS[request.engine].container_name
        requested_image = TRANSCRIPTION_ENGINE_CONFIGS[request.engine].docker_image

        # Container name attribute in Docker always has a leading slash
        if self.transcriber_container is not None and self.transcriber_container.attrs['Name'] != f'/{requested_container_name}':
            logger.debug(f'''Need to swap the current Docker container for a different one. 
            Current container: {self.transcriber_container.attrs['Name']}
            Requested container: {requested_container_name}
''')
            await self.stop_container(self.transcriber_container)
            self.transcriber_container = None

        if self.transcriber_container is None:
            logger.debug(f'Starting a new Docker container')
            self.transcriber_container = self.docker_client.containers.run(
                image=requested_image,
                detach=True,
                device_requests=[
                    {
                        "driver": "nvidia",
                        "count": -1,
                        "capabilities": [["gpu"]]
                    }
                ],
                environment={
                    'TRANSCRIBER_MODULE': TRANSCRIPTION_ENGINE_CONFIGS[request.engine].transcriber_module,
                    'NVIDIA_VISIBLE_DEVICES': 'all'
                },
                name=requested_container_name,
                remove=True,
                ports={'3001/tcp': 3001},
                runtime="nvidia",
                tty=True,
                volumes={
                    TRANSCRIPTION_ENGINE_CONFIGS[request.engine].download_path:
                        {'bind': '/project/app/transcribers/asr_models', 'mode': 'ro'},
                    self.input_dir:
                        {'bind': '/project/app/input', 'mode': 'ro'}
                }
            )
            logger.debug(f'New container: {self.transcriber_container}')
            logger.debug(f'Container attributes: {self.transcriber_container.attrs}')

        # Get default values if not provided
        if request.model_spec is None or request.model_spec.model_name is None:
            request.model_spec = ModelSpec(model_name=TRANSCRIPTION_ENGINE_CONFIGS[request.engine].default_model)


        # Wait until the service is ready
        transcriber_state = await self.wait_for_service(url='http://127.0.0.1:3001/transcription_service/state')
        if not transcriber_state.healthy:
            return TranscriptionResponse(
                transcript_text=None,
                transcript_segments=None,
                language_code=request.language_code,
                error=transcriber_state.details
            )

        # Simple timeout logic: time to load a model and transcription time, which is about 0.2 x audio duration
        duration = await self.audio_duration(request.filepath)
        timeout = 120  + 0.3 * duration

        # Transcription
        logger.info(f'Ready to start transcription')
        basename = os.path.basename(request.filepath)
        shutil.copyfile(request.filepath, os.path.join(self.input_dir, basename))
        new_request = TranscriptionServiceRequest(**request.model_dump())
        new_request.filepath = os.path.join('/project/app/input', basename)
        logger.info(f'Sending request for transcription: {new_request}')
        try:
            response = httpx.post(
                'http://127.0.0.1:3001/transcription_service/transcribe',
                json=new_request.model_dump(),
                timeout=timeout
            )
            response.raise_for_status()
            logger.info(f'Transcription finished')
            logger.debug(f'Response: {response.json()}')

            response = await self.segment_transcript(SegmentationRequest(segmentation=request.segmentation, **response.json()))
            logger.debug(f'Response from the transcript segmentation function: {response}')

            if request.save_to_file:
                await self._save_transcript(
                    transcription_output=response,
                    source_path=request.filepath,
                    transcript_formats=request.transcript_formats
                )
            return response
        except httpx.RequestError as exc:
            message = f'Transcription request failed with error: {exc}'
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )
        except httpx.HTTPStatusError as exc:
            message = f"Transcription request received an error response: {exc}"
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )
        except Exception as exc:
            message = f'Transcription failed with error" {str(exc)}'
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )


    async def _save_transcript(
            self,
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
            saving_path = os.path.join(self.output_dir, f'{filename_stem}.txt')
            with open(saving_path, 'w') as f:
                f.write(transcription_output.transcript_text)
            logger.debug(f'Transcript text saved at {saving_path}')
        if transcript_formats is None or TranscriptFormat.SRT in transcript_formats:
            saving_path = os.path.join(self.output_dir, f'{filename_stem}.srt')
            with open(saving_path, 'w') as f:
                f.write(segments2srt(transcription_output.transcript_segments))
            logger.debug(f'Transcript in the SRT format saved at {saving_path}')
        if transcript_formats is None or TranscriptFormat.JSON in transcript_formats:
            saving_path = os.path.join(self.output_dir, f'{filename_stem}.json')
            with open(saving_path, 'w') as f:
                output = [segment.model_dump() for segment in transcription_output.transcript_segments]
                json.dump(output, f)
            logger.debug(f'Transcript JSON saved at {saving_path}')

    @staticmethod
    async def audio_duration(filepath: FilePath) -> float | int:
        """Determine audio duration in seconds"""
        try:
            duration = librosa.get_duration(path=filepath)
            logger.debug(f'Record duration: {duration}')
            return duration
        except Exception as e:
            error_message = f'Failed to get audio duration with error: {str(e)}'
            logger.error(error_message)
            raise RuntimeError(error_message)

    async def stop_container(self, container):
        logger.debug(f'Stopping container {container.attrs['Name']}')
        try:
            container.stop()
            if not await self.wait_for_container_removal(container):
                error = 'Failed to remove a container'
                logger.error(error)
                raise RuntimeError(error)
        except Exception as e:
            error = f'Error while stopping a container: {str(e)}'
            logger.error(error)
            raise RuntimeError(error)

    @staticmethod
    async def wait_for_container_removal(container, interval=5, timeout=30):
        """Wait until Docker confirms the container is gone."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                container.reload()  # Will raise NotFound if removed
                logger.debug('Waiting for container removal. The container is still there')
                await asyncio.sleep(interval)
            except docker.errors.NotFound:
                logger.debug('Container removed')
                return True  # Container successfully removed
        logger.error('Container removal timeout has been reached')
        return False  # Timeout reached

    @staticmethod
    async def wait_for_service(url, interval=5, timeout=60) -> ServiceState:
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = httpx.get(url)
                logger.debug(f'Response from the health check endpoint: {response.json()}')
                if response.status_code == 200 and response.json()['healthy']:
                    return ServiceState(**response.json())
            except:
                pass
            await asyncio.sleep(interval)
        try:
            response = httpx.get(url).json()
            return ServiceState(**response)
        except Exception as e:
            return ServiceState(
                healthy=False,
                details=f'Final health test failed with error {str(e)}'
            )

    async def segment_transcript(self, request: SegmentationRequest) -> TranscriptionResponse:
        """Re-segment transcript as requested"""

        logger.debug(f'Request for segmentation received: {request}')

        if request.segmentation == SegmentationType.AUTO:
            logger.debug('Segmentation not required')
            return TranscriptionResponse(**request.model_dump())

        sentence_tokenizer_name = self.sentence_tokenizer_selector.get(request.language_code)
        if not sentence_tokenizer_name:
            error = (f'Transcript segmentation left unchanged: no sentence tokenizer available for the transcript '
                     f'language ({request.language_code})')
            logger.debug(error)
            response = TranscriptionResponse(**request.model_dump())
            response.error = error
            return response

        # Spin up a sentence tokenizer container for the transcript language if necessary
        if self.sentence_tokenizer_container is not None and self.sentence_tokenizer_container.attrs['Name'] \
                != f'/{SENTENCE_TOKENIZER_CONFIGS[sentence_tokenizer_name].container_name}':
            logger.debug(f'''Need to swap the current Docker container for a different one. 
                        Current container: {self.sentence_tokenizer_container.attrs['Name']}
                        Requested container: {sentence_tokenizer_name}
            ''')
            await self.stop_container(self.sentence_tokenizer_container)
            self.sentence_tokenizer_container = None

        if self.sentence_tokenizer_container is None:
            config = SENTENCE_TOKENIZER_CONFIGS[sentence_tokenizer_name]
            logger.debug(f'Starting a tokenizer container for language code: {request.language_code}')
            self.sentence_tokenizer_container = self.docker_client.containers.run(
                image=config.docker_image,
                detach=True,
                environment={
                    'SENTENCE_TOKENIZER_MODULE': config.tokenizer_module,
                },
                name=config.container_name,
                remove=True,
                ports={'3002/tcp': 3002},
                tty=True,
                volumes={
                    config.download_path:
                        {'bind': '/project/app/nlp/transcript_segmentation/tokenizer_models', 'mode': 'ro'},
                }
            )
            logger.debug(f'New container: {self.sentence_tokenizer_container}')
            logger.debug(f'Container attributes: {self.sentence_tokenizer_container.attrs}')

        # Wait until the service is ready
        tokenizer_state = await self.wait_for_service(url='http://127.0.0.1:3002/nlp/segmentation_service_state')
        if not tokenizer_state.healthy:
            response = TranscriptionResponse(**request.model_dump())
            response.error = tokenizer_state.details
            logger.error(f'Tokenizer not healthy. Details: {tokenizer_state.details}')
            return response

        logger.info(f'Sending request for transcript segmentation')

        # Primitive logic to define timeout
        duration = request.transcript_segments[-1].end
        segmentation_timeout = 0.25 * duration

        try:
            response = httpx.post(
                'http://127.0.0.1:3002/nlp/segmentation',
                json=request.model_dump(),
                timeout=segmentation_timeout
            )
            response.raise_for_status()
            logger.info(f'Segmentation finished')
            logger.debug(f'Response: {response.json()}')
            return TranscriptionResponse(**response.json())

        except httpx.RequestError as exc:
            message = f'Segmentation request failed with error: {exc}'
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )
        except httpx.HTTPStatusError as exc:
            message = f"Segmentation request received an error response: {exc}"
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )
        except Exception as exc:
            message = f'Segmentation failed with error" {str(exc)}'
            logger.error(message)
            return TranscriptionResponse(
                language_code=request.language_code,
                error=message
            )



