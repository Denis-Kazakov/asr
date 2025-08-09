import logging

from pydantic import Field, BaseModel, DirectoryPath

from .data_models import TranscriptionEngine


logger = logging.getLogger(__name__)


class TranscriptionEngineConfig(BaseModel):
    """Allowed parameters of a transcription engine"""
    models: list[str] = Field(
        ...,
        description='Available model sizes/checkpoints'
    )
    download_path: DirectoryPath | None = Field(
        default=None,
        description='Path to the folder with models on a local disk (will be bind mounted to a Docker container to '
                    'prevent redownloading)'
    )
    container_name: str = Field(..., description='Name of the container in which the engine will run')
    docker_image: str = Field(..., description='Docker image for the container')
    transcriber_module: str = Field(..., description='Transcriber module defining SpeechTranscriber class')
    default_model: str | None
    supports_gpu: bool = Field(default=True)
    word_timestamps: bool = Field(..., description='Support for word-level timestamps?')
    compute_types: list[str] | None = Field(default=None, description='Allowed compute types, e.g. float16')
    kwargs: dict | None = Field(default=None, description='Engine-specific parameters')


TRANSCRIPTION_ENGINE_CONFIGS = {
    TranscriptionEngine.WHISPER_LOCAL: TranscriptionEngineConfig(
        models=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
        download_path=DirectoryPath('/home/denis/Models/ASR/whisper/'),
        container_name='local_whisper',
        docker_image='local_whisper:0',
        transcriber_module='app.transcribers.local_whisper.src.transcriber',
        default_model="large-v3",
        supports_gpu=True,
        word_timestamps=True,
        compute_types=['fp16', 'fp32']
    ),
    TranscriptionEngine.FASTER_WHISPER: TranscriptionEngineConfig(
        models=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
        download_path=DirectoryPath('/home/denis/Models/ASR/faster-whisper/'),
        docker_image='faster_whisper:0',
        container_name='faster_whisper',
        transcriber_module='app.transcribers.faster_whisper.src.transcriber',
        default_model="medium",
        supports_gpu=True,
        word_timestamps=True,
        compute_types=['int8', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int16', 'float16', 'bfloat16',
                       'float32']
    ),
}
