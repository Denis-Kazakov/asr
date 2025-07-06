import string
from datetime import datetime, timedelta
import logging

import srt

from src.data_models import TranscriptSegment


logger = logging.getLogger(__name__)

class Normalizer:
    def __init__(self):
        self.exclude = string.punctuation + '\n'

    def _inverse_normalize_string(self, s: str):
        assert isinstance(s, str), "The argument must be a string"
        return s.translate(str.maketrans('', '', self.exclude)).lower()

    def inverse_normalize(self, s: str):
        """
        Remove punctuation and linebreaks from a string or list of strings and convert to lowercase.
        """
        if isinstance(s, str):
            return self._inverse_normalize_string(s)
        elif isinstance(s, list):
            return [self._inverse_normalize_string(item) for item in s]
        else:
            raise ValueError("The argument must be a string or list of strings")


def segments2srt(segments: list[TranscriptSegment]):
    """Convert a list of timestamped segments into the SRT format"""
    subtitle_list = []
    for n, segment in enumerate(segments):
        try:
            subtitle_list.append(srt.Subtitle(n,
                                              timedelta(seconds=segment.start),
                                              timedelta(seconds=segment.end),
                                              segment.text))
        except Exception as error:
            logger.error(f"Error: {error} with {segment}\n")

    return srt.compose(subtitle_list)
