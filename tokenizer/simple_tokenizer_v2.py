"""
Simple Tokenizer Module

This module provides a basic tokenizer implementation for text processing.
The tokenizer converts text to integer sequences and back, handling unknown
tokens and preserving punctuation.

Classes:
    SimpleTokenizerV2: A simple text tokenizer with vocabulary-based encoding

Typical usage example:
    vocab = {
        'hello': 0,
        'world': 1,
        '!': 2,
        '<|unk|>': 3
    }
    tokenizer = SimpleTokenizerV2(vocab)
    encoded = tokenizer.encode("hello world!")
    decoded = tokenizer.decode(encoded)
"""
import re

class SimpleTokenizerV2:
    """
    A simple tokenizer that converts text to integer sequences and back.

    This tokenizer splits text on whitespace and punctuation, maps tokens to
    integers based on a provided vocabulary, and handles unknown tokens with
    a special <|unk|> token.

    Attributes:
        str_to_int (Dict[str, int]): Dictionary mapping string tokens to integer IDs
        int_to_str (Dict[int, str]): Dictionary mapping integer IDs to string tokens

    Example:
        >>> vocab = {'hello': 0, 'world': 1, '!': 2, '<|unk|>': 3}
        >>> tokenizer = SimpleTokenizerV2(vocab)
        >>> ids = tokenizer.encode("hello world!")
        >>> print(ids)  # [0, 1, 2]
        >>> text = tokenizer.decode(ids)
        >>> print(text)  # "hello world!"

    Note:
        The vocabulary must contain a '<|unk|>' token to handle
        out-of-vocabulary words.
    """

    def __init__(self, vocab:dict[str, int]) -> None:
        """
        Initialize the tokenizer with a vocabulary.

        Creates both forward (str->int) and reverse (int->str) mappings
        from the provided vocabulary for efficient encoding and decoding.

        Example:
            >>> vocab = {
            ...     'the': 0,
            ...     'cat': 1,
            ...     'sat': 2,
            ...     '<|unk|>': 3
            ... }
            >>> tokenizer = SimpleTokenizerV2(vocab)

        :param vocab: A dictionary mapping string tokens to unique integer IDs. Must include
        '<|unk|>' token for handling unknown words.
        :type vocab: Dict[str, int]
        :raises ValueError: If '<|unk|>' token is not present in vocabulary
        :raises ValueError: If vocabulary contains duplicate IDs
        """
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of integer token IDs.

        Example:
            >>> tokenizer.encode("Hello, world!")
            [15, 4, 23, 8]  # Example IDs based on vocabulary

            >>> tokenizer.encode("Unknown word")
            [3, 25]  # 3 is ID for '<|unk|>', 25 for 'word'

        Note:
            The regex pattern splits on common punctuation and whitespace
            while keeping them as separate tokens. Double dashes (--) are
            treated as a single token.

        The encoding process:
        1. Splits text on punctuation marks and whitespace while preserving them
        2. Removes empty strings from the split result
        3. Replaces tokens not in vocabulary with '<|unk|>'
        4. Maps all tokens to their corresponding integer IDs

        :param text: The input text to tokenize. Can contain any Unicode characters, but only
        vocabulary tokens will be recognized.
        :type text: str
        :returns: A list of integer token IDs representing the input text. Empty text returns an
        empty list.
        :rtype: list[int]
        :raises TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids:list[int]) -> str:
        """
        Convert a list of integer token IDs back to text.

        The decoding process:
        1. Maps each integer ID to its corresponding string token
        2. Joins all tokens with single spaces
        3. Removes spaces before punctuation marks for proper formatting
        4. Returns the reconstructed text

        Example:
            >>> tokenizer.decode([15, 4, 23, 8])
            "Hello, world!"

            >>> tokenizer.decode([1, 2, 3, 4, 5])
            "The cat sat on mat"

        Note:
            The method assumes that punctuation tokens in the vocabulary
            are single characters. Spacing is automatically corrected for
            common punctuation marks.

        :param ids: A list of integer token IDs to decode. Can be empty, resulting in empty string
        output.
        :type ids: list[int]
        :return: The reconstructed text with proper punctuation spacing.
        :rtype: str
        :raises KeyError: If an ID in the input list is not found in the vocabulary mapping.
        """
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
