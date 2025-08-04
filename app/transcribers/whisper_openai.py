import json
import os
from typing import List, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from .transcriber_base import SpeechTranscriberBase
from speech2text.audio_chunker import Chunker


class OpenAIWhisper(SpeechTranscriberBase):
    """ASR and post-processing with OpenAI API"""

    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        with open('./speech2text/transcribers/whisper_languages.json') as f:
            self.lang_to_code = json.load(f)['lang_to_code']

    def __call__(self, full_path: str,
                 language: str,
                 return_timestamps: bool,
                 **kwargs) -> Tuple[str, List[dict]]:
        # Max file size is 25 Mb, but I use 24 for safety
        max_file_size = 24 * 2 ** 20
        # Files greater than 25 Mb need to be spit
        if os.path.getsize(full_path) > max_file_size:
            chunker = Chunker()
            chunk_paths = chunker.split(full_path)  # chunk_paths is an iterator
            text = ''
            segments = []
            for chunk_path in chunk_paths:
                chunk_text, chunk_segments = self._transcribe(chunk_path, language, return_timestamps)
                text += chunk_text
                segments.append(chunk_segments)
            if return_timestamps:
                segments = chunker.merge(segments)
            else:
                segments = [None]
        else:
            text, segments = self._transcribe(full_path, language, return_timestamps)
        return text, segments

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        reraise=True
    )
    def _transcribe(self, full_path: str, language: str, return_timestamps: bool):
        """Transcribe a single audio piece"""
        if language:
            language_code = self.lang_to_code[language]  # OpenAI requires ISO language code, not full language name
        else:
            language_code = None

        if language in self.convert2sentences.lang_to_tokenizer:
            granularity = 'word'
        else:
            granularity = 'segment'

        if return_timestamps:
            with open(full_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_code,
                    response_format='verbose_json',
                    timestamp_granularities=[granularity]
                ).to_dict()
            text = transcript['text']
            detected_language = transcript['language']
            if language in self.convert2sentences.lang_to_tokenizer:
                words = transcript['words']
                segments = self.convert2sentences(text, words, detected_language)
            else:
                segments = transcript['segments']
            return text, segments
        else:
            with open(full_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_code,
                    response_format='text'
                )
            return transcript, [None]
