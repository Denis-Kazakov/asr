from email.policy import default
from enum import Enum
# from enum import StrEnum   # Will not work in the Faster Whisper container which uses Python 3.10
from typing import Any, Literal
import logging

from pydantic import Field, BaseModel, FilePath


logger = logging.getLogger(__name__)


class TranscriptionEngine(str, Enum):
    """Transcription engines"""
    WHISPER_LOCAL = "whisper_local"
    # WHISPER_HUGGING_FACE = "whisper_hf"
    FASTER_WHISPER = "faster_whisper"
    # WHISPER_CPP = "whisper_cpp"


class TranscriptFormat(str, Enum):
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
    device: str | None = Field(default='cuda', description='Device (cpu/cuda)')
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
    engine: TranscriptionEngine | None = TranscriptionEngine.FASTER_WHISPER
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

class TranscriptionRequestForm(BaseModel):
    """Simple request from an HTML form"""
    filepath: FilePath = Field(..., description='Path to a media file')
    language_code: str | None = Field(default=None, description='ISO language code, e.g. "en"')
    engine: TranscriptionEngine | None = TranscriptionEngine.FASTER_WHISPER
    model_name: str = Field(..., description="ASR model to be used, e.g. 'medium'")


class TranscriptionServiceState(BaseModel):
    """State of a transcription service container"""
    healthy: bool = Field(..., description='Is a model loaded and the transcriber ready to receive requests?')
    details: str = Field(..., description='Error message or loaded model specifications')


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
