import logging
from enum import Enum  # StrEnum will not work in the Faster Whisper container which uses Python 3.10
from typing import Any

from pydantic import Field, BaseModel, FilePath, model_validator
from typing_extensions import Self

from app.shared.model_configs import TranscriptionEngine, TRANSCRIPTION_ENGINE_CONFIGS

logger = logging.getLogger(__name__)

class SegmentationType(str, Enum):
    """Types of timestamped transcript segmentation"""

    AUTO = 'auto'
    """As returned by the model"""

    SENTENCE = 'sentence'
    """Rearrange the transcript into timestamped sentences"""


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
    segmentation: SegmentationType | None = SegmentationType.AUTO

    save_to_file: bool | None = Field(
        default=True,
        description='Save transcripts to files with the same name but different extensions'
    )

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        # Check that the model name can be used with the engine
        if self.model_spec.model_name not in TRANSCRIPTION_ENGINE_CONFIGS[self.engine].models:
            error = f'{self.engine} cannot use model {self.model_spec.model_name}'
            logger.error(error)
            raise ValueError(error)

        # Check that the model supports word-level timestamps if they are requested
        if self.word_timestamps and not TRANSCRIPTION_ENGINE_CONFIGS[self.engine].word_timestamps:
            error = f'{self.engine} does not support word-level timestamps'
            logger.error(error)
            raise ValueError(error)

        # Check that the model supports word-level timestamps if segmentation by sentence is requested
        if self.segmentation == 'sentence' and not TRANSCRIPTION_ENGINE_CONFIGS[self.engine].word_timestamps:
            error = f'{self.engine} does not support word-level timestamps so segmentation by sentence is not possible'
            logger.error(error)
            raise ValueError(error)

        return self


class TranscriptionRequestForm(BaseModel):
    """Simple request from an HTML form"""
    filepath: FilePath = Field(..., description='Path to a media file')
    language_code: str | None = Field(default=None, description='ISO language code, e.g. "en"')
    engine: TranscriptionEngine | None = TranscriptionEngine.FASTER_WHISPER
    model_name: str = Field(..., description="ASR model to be used, e.g. 'medium'")


class ServiceState(BaseModel):
    """State of a transcription service container"""
    healthy: bool = Field(..., description='Is a model loaded and the service ready to receive requests?')
    details: str = Field(..., description='Error message or loaded model specifications')


class TimeStampedWord(BaseModel):
    start: float = Field(..., description='Start time of the word', ge=0)
    end: float = Field(..., description='End time of the word', ge=0)
    word: str = Field(..., description='The word')


class TranscriptSegment(BaseModel):
    start: float = Field(..., description='Start time of the segment', ge=0)
    end: float = Field(..., description='End time of the segment', ge=0)
    text: str = Field(..., description='Segment text')
    words: list[TimeStampedWord] | None = Field(default=None, description='Words included in this segment with their timestamps')


class TranscriptionResponse(BaseModel):
    transcript_text: str | None = Field(
        default=None,
        description='Transcript as plain text'
    )
    transcript_segments: list[TranscriptSegment] | None = None
    language_code: str | None = Field(..., description='ISO language code, e.g. "en"')
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

class SegmentationRequest(TranscriptionResponse):
    """Request to change transcript segmentation, e.g. make sure it is segmented by sentences"""
    segmentation: SegmentationType
