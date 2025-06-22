from enum import StrEnum
from typing import Union

from pydantic import Field, BaseModel, FilePath, conlist


class TranscriptionEngine(StrEnum):
    """Engine name values should be the same as module names"""
    WHISPER_LOCAL = "whisper_local"
    #WHISPER_HUGGING_FACE = "whisper_hf"
    #FASTER_WHISPER = "faster_whisper"
    #WHISPER_CPP = "whisper_cpp"

class TranscriptionEngineConfig(BaseModel):
    """Allowed parameters of a transcription method"""
    models: dict[str, Union[FilePath, None]] = Field(..., description='Existing models with paths to pre-downloaded files')
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
    # ...
}

class TranscriptFormat(StrEnum):
    """Transcript format"""
    TXT = 'txt'
    """Plain text, no timestamps, no paragraph breaks"""

    SRT = 'srt'
    """SRT format with sequentially numbered and timestamped segments output by the model"""

    SRT_SENTENCES = 'srt_sentences'
    """Same as SRT but the transcript is re-segmented so that each segment is a sentence"""


class TranscriptionRequest(BaseModel):
    filepath: FilePath = Field(..., description='Path to a media file')
    language_code: str | None = Field(default=None, description='ISO language code, e.g. "en"')
    engine: TranscriptionEngine = TranscriptionEngine.WHISPER_LOCAL
    model_name: str | None = Field(default=None, description="ASR model to be used")
    transcript_formats: list[TranscriptFormat] | None = Field(
        default=None,
        description='Transcript format, e.g. SRT'
    )
    compute_type: str | None = Field(default=None, description="Compute type such as fp16")

class TranscriptionResponse(BaseModel):
    transcripts: conlist(FilePath) | None = Field(
        ...,
        description='Paths to one or more transcript files in specified formats'
    )
    error: str | None = Field(default=None, description='Error message if transcription failed')

