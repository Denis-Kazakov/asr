from enum import StrEnum
from typing import Any, Literal
import logging

from pydantic import Field, BaseModel, FilePath, DirectoryPath

logger = logging.getLogger(__name__)


class TranscriptionEngine(StrEnum):
    """Transcription engines"""
    WHISPER_LOCAL = "whisper_local"
    # WHISPER_HUGGING_FACE = "whisper_hf"
    FASTER_WHISPER = "faster_whisper"
    # WHISPER_CPP = "whisper_cpp"


class TranscriptionEngineConfig(BaseModel):
    """Allowed parameters of a transcription engine"""
    models: list[str] = Field(
        ...,
        description='Available model sizes/checkpoints'
    )
    base_path: DirectoryPath | None = Field(
        default=None,
        description='Path to the folder with models on a local disk (will be bind mounted to a Docker container to '
                    'prevent redownloading)'
    )
    container_name: str = Field(..., description='Name of the container in which the engine will run')
    docker_image: str = Field(..., description='Docker image for the container')
    default_model: str | None
    supports_gpu: bool = Field(default=True)
    word_timestamps: bool = Field(..., description='Support for word-level timestamps?')
    compute_types: list[str] | None = Field(default=None, description='Allowed compute types, e.g. float16')
    kwargs: dict | None = Field(default=None, description='Engine-specific parameters')


TRANSCRIPTION_ENGINE_CONFIGS = {
    TranscriptionEngine.WHISPER_LOCAL: TranscriptionEngineConfig(
        models=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
        base_path=DirectoryPath('/home/denis/Models/ASR/whisper/'),
        container_name='local_whisper',
        docker_image='local_whisper',
        default_model="large-v3",
        supports_gpu=True,
        word_timestamps=True,
        compute_types=['fp16', 'fp32']
    ),
    TranscriptionEngine.FASTER_WHISPER: TranscriptionEngineConfig(
        models=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
        base_path=DirectoryPath('/home/denis/Models/ASR/faster-whisper/'),
        docker_image='faster_whisper',
        container_name='faster_whisper',
        default_model="medium",
        supports_gpu=True,
        word_timestamps=True,
        compute_types=['int8', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int16', 'float16', 'bfloat16',
                       'float32']
    ),
}


class TranscriptFormat(StrEnum):
    """Transcript format"""
    TXT = 'txt'
    """Plain text, no timestamps, no paragraph breaks"""

    SRT = 'srt'
    """SRT format with sequentially numbered and timestamped segments output by the model"""

    JSON = 'json'
    """JSON with timestamped segments"""


class ModelSpec(BaseModel):
    """Specifications for a model to be loaded into a transcriber"""
    model_name: str = Field(..., description="ASR model to be used, e.g. 'medium'")
    model_path: str | None = Field(default=None, description='Root directory where models can be saved in advance')
    device: str | None = Field(default=None, description='Device (cpu/cuda)')
    model_kwargs: dict[str, Any] | None = Field(
        default={},
        description='Engine-specific model parameters such as compute type (e.g. fp16)'
    )


class TranscriptionServiceRequest(BaseModel):
    """Request from the gateway to a transcription service"""
    filepath: FilePath = Field(..., description='Path to a media file')
    language_code: str | None = Field(default=None, description='ISO language code, e.g. "en"')
    model_spec: ModelSpec | None = Field(default=None, description="Specifications for a model to be used")
    word_timestamps: bool = Field(default=False, description='Return word-level timestamps')
    transcription_kwargs: dict[str, Any] | None = Field(
        default={},
        description='Engine-specific transcription parameters such as temperature'
    )


class TranscriptionRequest(TranscriptionServiceRequest):
    """External transcription request to the gateway"""
    engine: TranscriptionEngine = TranscriptionEngine.WHISPER_LOCAL
    transcript_formats: list[TranscriptFormat] | None = Field(
        default=None,
        description='Format of the transcript returned to the user, e.g. SRT'
    )
    segmentation: Literal['auto', 'sentence'] = Field(
        default='auto',
        description=('How the transcript should be split into time-stamped segments: auto (as returned by the model)'
                     'or sentence (rearrange the transcript into time-stamped segments')
    )
    save_to_file: bool | None = Field(
        default=True,
        description='Save transcripts to files with the same name but different extension'
    )

    # @field_validator('model_name')
    # @classmethod
    # def validate_model(cls, model_name: str):
    #     if model_name is not None and model_name not in TRANSCRIPTION_ENGINE_CONFIGS[cls.engine].models.keys():
    #         error = f'{cls.engine} does not have model {model_name}'
    #         logger.error(error)
    #         raise ValueError(error)


class TranscriptSegment(BaseModel):
    start: float = Field(..., description='Start time of a segment', ge=0)
    end: float = Field(..., description='End time of a segment', ge=0)
    text: str = Field(..., description='Segment text')


class TranscriptionResponse(BaseModel):
    transcript_text: str | None = Field(
        default=None,
        description='Transcript as a plain text'
    )
    transcript_segments: list[TranscriptSegment] | None = None
    error: str | None = Field(default=None, description='Error message if transcription failed')


class TranscriberState(BaseModel):
    """Current state of a transcriber"""
    model_name: str | None = Field(default=None, description="Name of the loaded model (checkpoint), e.g. 'medium'")
    busy: bool = Field(default=False, description="Are there any incomplete jobs with the model?")


class CalculateASRMetricsRequest(BaseModel):
    hypothesis: str | list[str]
    reference: str | list[str]


class TranscribeCalculateASRMetricsRequest(TranscriptionRequest):
    reference: str


class ASRMetricsResponse(BaseModel):
    wer: float = Field(..., description='Word error rate', ge=0)
    cer: float = Field(..., description='Character error rate', ge=0)
    wer_normalized: float = Field(..., description='Word error rate after inverse normalization', ge=0)
    cer_normalized: float = Field(..., description='Character error rate after inverse normalization', ge=0)
