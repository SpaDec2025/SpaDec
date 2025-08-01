import os
from logging import getLogger
from typing import List
from transformers import AutoTokenizer

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        if not os.path.exists(model_path):
            self.sp_model = AutoTokenizer.from_pretrained(os.path.dirname(model_path), trust_remote_code=True)
            if "Qwen" in model_path:
                self.n_words: int = 152064
                self.bos_id: int = self.sp_model.bos_token_id
                self.eos_id: int = self.sp_model.eos_token_id
                self.pad_id: int = self.sp_model.pad_token_id
            logger.info(f"AutoTokenizer.from_pretrained() from {os.path.dirname(model_path)}")
        else:
            assert os.path.isfile(model_path), model_path
            self.sp_model = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")
            # BOS / EOS token IDs
            self.n_words: int = self.sp_model.vocab_size()
            self.bos_id: int = self.sp_model.bos_id()
            self.eos_id: int = self.sp_model.eos_id()
            self.pad_id: int = self.sp_model.pad_id()
            assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}") # [Modified] print

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)