import logging
from typing import List, Union

import stanza

from app.nlp.transcript_segmentation.segmenter_base import SegmenterBase
from app.shared.data_models import SegmentationRequest, TranscriptionResponse

logger = logging.getLogger('default')

class Segmenter(SegmenterBase):
    """
    Convert a list of segments into a list of sentences
    Speech-to-text models split text into segments at random points, so the purpose is to reformate the transcript
    so that each segment is a sentence.
    """
    def __init__(self):
        """
        self.splitters: dictionary of Stanza pipelines for splitting, one for each language
        """
        self.splitters = {}
        super().__init__()

    async def __call__(self, transcript: SegmentationRequest) -> TranscriptionResponse:
        """
        Sentence tokenization of a speech transcript
        Inputs:
            transcript_segments: list of segments with each segment having start and end timestamps and text as well as
                the start and end timestamps of each word.
        Outputs:
            List of segments with the same format but each segment is now a sentence.
        """
        logger.debug(f'Segmentation service received request for segmentation: {transcript}')
        # Get a list of timestamped words
        words = []
        for segment in transcript.transcript_segments:
            words.extend([dict(word) for word in segment.words])
        if not words:
            raise ValueError('No time-stamped words in the transcript')

        # Concatenate transcript words into a single text with known positions of each word start and end
        current_position = 0
        concatenated_words = ''  # Temporary text with known position of each word's first and last character
        for word in words:
            token = word['word'].strip()
            word['start_position'] = current_position
            word['end_position'] = current_position + len(token)
            concatenated_words += token + ' '
            current_position = len(concatenated_words)
        assert current_position == len(concatenated_words), "Transcript word concatenation error"

        # Check if Stanza model for the language has been loaded and load if not
        if not self.splitters.get(transcript.language):
            logger.info(f'Loading Stanza model for language code: {transcript.language}')
            try:
                self.splitters[transcript.language] = stanza.Pipeline(transcript.language, processors='tokenize')
            except Exception as e:
                logger.error(f'Failed to load Stanza model for language code: {transcript.language}. Error: {str(e)}')
                raise ImportError(f'Model for language code {transcript.language} cannot be loaded. Error: {str(e)}')

        # Split transcript text into sentences
        logger.debug(f'Text for splitting: {transcript.text}')
        sentences = [sentence.text for sentence in self.splitters[transcript.language](transcript.text).sentences]
        logger.debug(f'Sentences after splitting: {sentences[:5]}')

        # Concatenate sentences into a single text and get sentence number for each character.
        # (Character positions can differ in different concatenations due to different numbers of whitespace characters.)
        sentence_numbers = []  # Sentence number of each character in the sentence splitter output
        concatenated_sentences = ''
        for n, sentence in enumerate(sentences):
            sentence_numbers += [n] * (len(sentence) + 1)
            concatenated_sentences += sentence + ' '
        assert len(sentence_numbers) == len(concatenated_sentences), "Sentence concatenation error"

        # Use dynamic programming (longest common subsequence) for matching
        s1 = concatenated_words
        s2 = concatenated_sentences
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i - 1] == s2[j - 1]:  # Characters match
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:  # Take the maximum value from the previous row or column
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        i = len(s1)
        j = len(s2)
        self.match = {}  # Dictionary matching character position in concatenated_words to sentence number
        while i > 0 and j > 0:
            if (dp[i][j] == dp[i - 1][j - 1] + 1) and (s2[j - 1] == s1[i - 1]):
                self.match[i - 1] = j - 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j]:
                i -= 1
            else:
                j -= 1

        # New segments to be included in the function output
        segments = []
        for sentence in sentences:
            segments.append(
                TranscriptSegment(
                    start=0.0,  # Value to initialize class instance, will be replaced with actual time
                    text=sentence,
                    words=[]
                )
            )

        # Add word data (word with timestamps) to each segment:
        for word in words:
            word_sentence = self._match_word_to_sentence(word, sentence_numbers)
            # Do not include character positions, only timestamps
            segments[word_sentence].words.append(
                TimestampedWord(
                    word= word['word'],
                    start=word['start'],
                    end=word['end']
                )
            )

        # Add sentence start and end timestamps: start of the first word and end of the last
        for segment in segments:
            segment.start = segment.words[0].start
            segment.end = segment.words[-1].end
        logger.debug(f'Segments: {segments[:3]}')
        return TranscriberOutput(
            language=transcript.language,
            text=transcript.text,
            segments=TranscriptSegments(segments=segments),
            duration=transcript.duration
        )

    def _get_sentence_number(self, s: List[int], idx: Union[int, None]) -> Union[int, None]:
        """
        Input: character number in concatenated_sentences (can be None)
        Output: number of the sentence containing this character
        """
        if idx is not None:
            try:
                return s[idx]
            except IndexError:
                return None
        else:
            return None

    def _match_word_to_sentence(self, word, sentence_numbers) -> int:
        """
        Get sentence number for each word of the original transcript.
        Hypothetically complex words can be divided between sentences. Also, there is a chance that some characters in
        the word will not be matched to a sentence. In such cases, the word will be included in the first sentence:
        the sentence of the first character that was matched to a sentence.
        """
        for i in range(word['start_position'], word['end_position'] + 1):
            sentence_number = self._get_sentence_number(sentence_numbers, self.match.get(i))
            if sentence_number is not None:
                return sentence_number
        raise RuntimeError(f'Word {word["word"]} not matched to any sentence.')
