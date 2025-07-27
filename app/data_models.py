from enum import StrEnum
from typing import Union, Any, Literal
import logging

from pydantic import Field, BaseModel, FilePath, DirectoryPath


logger = logging.getLogger(__name__)

class TranscriptionEngine(StrEnum):
    """Engine name values should be the same as module names"""
    WHISPER_LOCAL = "whisper_local"
    # WHISPER_HUGGING_FACE = "whisper_hf"
    FASTER_WHISPER = "faster_whisper"
    # WHISPER_CPP = "whisper_cpp"

class TranscriptionEngineConfig(BaseModel):
    """Allowed parameters of a transcription method"""
    models: dict[str, Union[FilePath, DirectoryPath, None]] = Field(
        ...,
        description='Existing models with paths to pre-downloaded files'
    )
    default_model: str | None
    supports_gpu: bool
    word_timestamps: bool = Field(..., description='Support for word-level timestamps?')
    compute_types: list[str] | None = Field(default=None, description='Allowed compute types, e.g. float16')
    transcriber_class: str = Field(..., description='Name of a class to be imported from the module, e.g. "LocalWhisper"')


TRANSCRIPTION_ENGINE_CONFIGS = {
    TranscriptionEngine.WHISPER_LOCAL: TranscriptionEngineConfig(
        models={
            'tiny.en': None,
            'tiny': '/home/denis/Models/ASR/whisper/tiny.pt',
            'base.en': None,
            'base': None,
            'small.en': None,
            'small': None,
            'medium.en': None,
            'medium': '/home/denis/Models/ASR/whisper/medium.pt',
            'large-v1': None,
            'large-v2': None,
            'large-v3': '/home/denis/Models/ASR/whisper/large-v3.pt',
            'large': None,
            'large-v3-turbo': '/home/denis/Models/ASR/whisper/large-v3-turbo.pt',
            'turbo': None
        },
        default_model="medium",
        supports_gpu=True,
        word_timestamps=True,
        transcriber_class='LocalWhisper',
        compute_types=['fp16', 'fp32']
    ),
    TranscriptionEngine.FASTER_WHISPER: TranscriptionEngineConfig(
        models={
            'tiny.en': '/home/denis/Models/ASR/faster-whisper/',
            'tiny': '/home/denis/Models/ASR/faster-whisper/',
            'base.en': '/home/denis/Models/ASR/faster-whisper/',
            'base': '/home/denis/Models/ASR/faster-whisper/',
            'small.en': '/home/denis/Models/ASR/faster-whisper/',
            'small': '/home/denis/Models/ASR/faster-whisper/',
            'medium.en': '/home/denis/Models/ASR/faster-whisper/',
            'medium': '/home/denis/Models/ASR/faster-whisper/',
            'large-v1': '/home/denis/Models/ASR/faster-whisper/',
            'large-v2': '/home/denis/Models/ASR/faster-whisper/',
            'large-v3': '/home/denis/Models/ASR/faster-whisper/',
            'large': '/home/denis/Models/ASR/faster-whisper/',
            'large-v3-turbo': '/home/denis/Models/ASR/faster-whisper/',
            'turbo': '/home/denis/Models/ASR/faster-whisper/'
        },
        default_model="medium",
        supports_gpu=True,
        word_timestamps=True,
        transcriber_class='FasterWhisper',
        compute_types=['int8', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int16', 'float16', 'bfloat16', 'float32']
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


class TranscriptionServiceRequest(BaseModel):
    """Request from the gateway to a transcription service"""
    filepath: FilePath = Field(..., description='Path to a media file')
    language_code: str | None = Field(default=None, description='ISO language code, e.g. "en"')
    model_name: str | None = Field(default=None, description="ASR model to be used")
    model_kwargs: dict[str, Any] | None = Field(
        default={},
        description='Engine-specific model parameters such as compute type (e.g. fp16)'
    )
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