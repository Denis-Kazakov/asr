from typing import List, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .transcriber_base import SpeechTranscriber


class LocalWhisperHF(SpeechTranscriber):
    """
    ASR with a local Whisper model in the HuggingFace implementation
    """

    def __init__(self, model_path: str):
        super().__init__()
        accelerator = Accelerator()
        self.device = accelerator.device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self,
                 full_path: str,
                 language: str,
                 return_timestamps: bool,
                 **kwargs) -> Tuple[str, List[dict]]:

        # HuggingFace transcriber has higher memory requirements for word-level timestamps,
        # which can be offset by reducing the batch size
        batch_size = 1 if return_timestamps else 4

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            max_new_tokens=128,
            chunk_length_s=10,
            batch_size=batch_size,
            device=self.device,
        )
        # Check whether it is possible to split into sentences:
        if language in self.convert2sentences.lang_to_tokenizer:
            sent_tokenizer_available = True
        else:
            sent_tokenizer_available = False

        if return_timestamps and sent_tokenizer_available:
            return_timestamps = 'word'  # Word-level granularity needed for merging into sentences
        result = pipe(
            full_path,
            return_timestamps=return_timestamps,
            generate_kwargs={"language": language}
        )
        torch.cuda.empty_cache()
        if return_timestamps:
            # Convert from the HF format into the local Whisper format:
            if sent_tokenizer_available:
                seg = 'word'
            else:
                seg = 'text'
            segments = []
            for chunk in result['chunks']:
                start, end = chunk['timestamp']
                if end is None:  # Can be None in the last segment
                    end = start + 0.01  # Adding arbitrary number
                segments.append(
                    {'start': start,
                     'end': end,
                     seg: chunk['text']}
                )
            if sent_tokenizer_available:
                segments = self.convert2sentences(
                    text=result['text'],
                    words=segments,
                    language=language
                )
        else:
            segments = [None]
        return result['text'], segments
