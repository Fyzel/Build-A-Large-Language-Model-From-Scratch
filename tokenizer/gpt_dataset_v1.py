"""
Module for GPT dataset implementation.

This module provides a PyTorch Dataset class for training GPT models by creating
input-target token pairs from text using a sliding window approach. The dataset
implements next-token prediction by pairing each sequence with its shifted version.
"""

from typing import Any, Tuple

import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """
        A PyTorch Dataset for GPT model training using sliding window tokenization.

        This dataset creates input-target token pairs by sliding a window over
        tokenized text. Each sample consists of a sequence of tokens as input
        and the next token sequence (shifted by one position) as target for
        next-token prediction training.

        The dataset uses a stride parameter to control overlap between consecutive
        samples, allowing for efficient use of training data while maintaining
        sequence continuity.

        Example:
            >>> # Assuming you have a tokenizer (e.g., from transformers library)
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> text = "The quick brown fox jumps over the lazy dog."
            >>> dataset = GPTDatasetV1(text, tokenizer, max_length=10, stride=5)
            >>> print(f"Dataset size: {len(dataset)}")
            >>> input_tokens, target_tokens = dataset[0]
            >>> print(f"Input shape: {input_tokens.shape}")
            >>> print(f"Target shape: {target_tokens.shape}")
        """

    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int) -> None:
        """
        Initialize the GPT dataset with text and tokenization parameters.

        Creates input-target token pairs by sliding a window of size max_length
        over the tokenized text with the specified stride. Each input sequence
        is paired with its corresponding target sequence (shifted by one token).

        :param txt: The input text to tokenize and create training samples from
        :type txt: str
        :param tokenizer: Tokenizer instance that must have an 'encode' method
        :type tokenizer: Any
        :param max_length: Maximum length of each token sequence (both input and target)
        :type max_length: int
        :param stride: Step size for sliding window over tokens (controls overlap)
        :type stride: int
        :raises AttributeError: If tokenizer doesn't have an 'encode' method
        :raises ValueError: If max_length is less than 1 or stride is less than 1
        :raises TypeError: If txt is not a string or max_length/stride are not integers

        Example:
            >>> dataset = GPTDatasetV1("Hello world!", tokenizer, max_length=5, stride=2)
        """
        self.input_ids = []
        self.target_ids = []

        if not hasattr(tokenizer, 'encode'):
            raise AttributeError("Tokenizer must have an 'encode' method")
        if not isinstance(txt, str):
            raise TypeError("Input text must be a string")
        if not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if not isinstance(stride, int):
            raise TypeError("stride must be an integer")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        :rtype: int
        :return: Number of input-target token pairs in the dataset

        Example:
            >>> len(dataset)
            42
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Retrieve a sample from the dataset by index.

                Returns the input token sequence and its corresponding target sequence
                (shifted by one position) as PyTorch tensors.

                :param idx: Index of the sample to retrieve (0-based)
                :type idx: int
                :rtype: Tuple[torch.Tensor, torch.Tensor]
                :return: Tuple containing (input_tokens, target_tokens) as tensors
                :raises IndexError: If idx is negative or >= len(dataset)
                :raises TypeError: If idx is not an integer

                Example:
                    >>> input_tokens, target_tokens = dataset[0]
                    >>> print(input_tokens.shape)   # torch.Size([max_length])
                    >>> print(target_tokens.shape)  # torch.Size([max_length])
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        if idx >= len(self.input_ids) or idx >= len(self.target_ids):
            raise IndexError("Index out of bounds for input or target IDs")

        return self.input_ids[idx], self.target_ids[idx]
