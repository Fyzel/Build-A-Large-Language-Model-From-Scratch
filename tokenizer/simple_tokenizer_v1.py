"""
Simple Text Tokenization Module

This module provides a basic tokenizer implementation for converting text into
numerical token IDs and vice versa. The SimpleTokenizerV1 class offers encoding
and decoding functionality using a predefined vocabulary.

Classes:
    SimpleTokenizerV1: A simple tokenizer that splits text on spaces and punctuation
                      and converts between token strings and their numerical IDs.

Usage:
    Instantiate SimpleTokenizerV1 with a vocabulary dictionary mapping tokens to IDs,
    then use the encode/decode methods to convert between text and token IDs.
"""
import re


class SimpleTokenizerV1:
    """
    Simple tokenizer that encodes text into token IDs and decodes token IDs back to text.
    """

    def __init__(self, vocab: dict):
        """
        Initialize the tokenizer with a vocabulary.

        Creates a bidirectional mapping between string tokens and their integer IDs.
        The tokenizer maintains both a forward mapping (string to int) for encoding
        and a reverse mapping (int to string) for decoding operations.

        :param vocab: A dictionary mapping token strings to integer IDs. Each key
                     should be a unique token string, and each value should be a
                     unique integer identifier. The vocabulary should include all
                     tokens that will be encountered during encoding/decoding.
        :type vocab: Dict[str, int]
        :raises ValueError: If the vocabulary contains duplicate values (non-unique IDs).
        :raises TypeError: If vocab is not a dictionary or contains non-string keys
                          or non-integer values.

        Example:
            >>> vocab = {"hello": 0, "world": 1, "!": 2}
            >>> tokenizer = SimpleTokenizerV1(vocab)
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.

        The method preprocesses the input text by splitting it on whitespace and
        punctuation marks, then maps each token to its corresponding ID from the
        vocabulary. Tokens are split using regex pattern that captures common
        punctuation and whitespace.

        :param text: Input text to be tokenized.
        :type text: str
        :return: List of token IDs corresponding to the input text.
        :rtype: List[int]
        :raises KeyError: If a token in the text is not found in the vocabulary.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token IDs back to text.

        The method maps each token ID to its corresponding string token using the
        vocabulary, joins them with spaces, and then applies post-processing to
        remove unwanted spaces before punctuation marks to restore natural text
        formatting.

        :param ids: List of token IDs to be converted back to text.
        :type ids: List[int]
        :return: Decoded text string with proper punctuation spacing.
        :rtype: str
        :raises KeyError: If a token ID is not found in the vocabulary.
        """
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
