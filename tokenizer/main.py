"""
Text Retrieval and Analysis Module

This module downloads 'the-verdict.txt' from a GitHub repository and performs
basic text analysis. It retrieves the text file using urllib, reads its
content, and outputs character count statistics and a preview of the text.

Functions:
    No explicit functions defined, functionality is executed at module level.

Usage:
    Run this module directly to download and analyze the text file.
"""

import re
import urllib.request

# from importlib.metadata import version
import tiktoken
from torch.utils.data import DataLoader

from tokenizer.gpt_dataset_v1 import GPTDatasetV1


def create_dataloader_v1(txt,
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0) -> DataLoader:
    """
    Create a DataLoader for the GPTDatasetV1.

    :param txt: The input text to be tokenized and processed.
    :type txt: str
    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param max_length: Maximum length of each sequence.
    :type max_length: int
    :param stride: Stride for overlapping sequences.
    :type stride: int
    :param shuffle: Whether to shuffle the dataset.
    :type shuffle: bool
    :param drop_last: Whether to drop the last incomplete batch.
    :type drop_last: bool
    :param num_workers: Number of subprocesses to use for data loading.
    :type num_workers: int
    :return: DataLoader instance for the GPTDatasetV1.
    :rtype: DataLoader
    :raises TypeError: If txt is not a string or shuffle/drop_last are not booleans.
    :raises ValueError: If batch_size, max_length, or stride are not positive integers.
    """

    if not isinstance(txt, str):
        raise TypeError('txt must be a string')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('stride must be a positive integer')
    if not isinstance(shuffle, bool):
        raise TypeError('shuffle must be a boolean')
    if not isinstance(drop_last, bool):
        raise TypeError('drop_last must be a boolean')
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a non-negative integer')

    tiktokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tiktokenizer, max_length, stride)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    print("Downloading 'the-verdict.txt', Edith Wharton’s short story, from GitHub...")

    URL = ('https://raw.githubusercontent.com/rasbt/'
           'LLMs-from-scratch/main/ch02/01_main-chapter-code/'
           'the-verdict.txt')
    FILE_PATH = 'the-verdict.txt'
    urllib.request.urlretrieve(URL, FILE_PATH)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print('Total number of character:', len(raw_text))
    print(raw_text[:99])

    print('\nTokenize the text into words and punctuation marks:')
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    print(preprocessed)

    print('\nRemove empty strings from the list:')
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed)

    print('\nTotal number of tokens:', len(preprocessed))

    print('\nFirst 30 tokens:')
    print(preprocessed[:30])

    print('\n----------------------------------------')
    print('Create the sorted vocabulary')
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print('\nVocabulary size:', vocab_size)

    print('\nFirst 50 tokens in the vocabulary:')
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)

        # Limit the output to the first 50 items
        if i >= 50:
            break

    print('\nSorting the vocabulary by token ID and extending to include |unk| and |endoftext|:')
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(['<|endoftext|>', '<|unk|>'])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))

    print('\nThe last five items in the vocabulary:')
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    tokenizer = tiktoken.get_encoding('gpt2')

    TEXT = (
        'Hello, do you like tea? <|endoftext|> In the sunlit terraces'
        ' of someunknownPlace.'
    )

    print('\nTokenizing the text using the GPT-2 tokenizer:')
    print(TEXT)

    integers = tokenizer.encode(TEXT, allowed_special={'<|endoftext|>'})

    print('\nTokenized text into integers:')
    print(integers)

    print('\nDecoding the integers back to strings:')
    strings = tokenizer.decode(integers)
    print(strings)

    TEXT = (
        'Akwirw ier'
    )

    print('\nTokenizing the text using the GPT-2 tokenizer:')
    print(TEXT)

    integers = tokenizer.encode(TEXT, allowed_special={'<|endoftext|>'})

    print('\nTokenized text into integers:')
    print(integers)

    print('\nDecoding the integers back to strings:')
    strings = tokenizer.decode(integers)
    print(strings)

    print('\n----------------------------------------')
    print('Use tiktoken to tokenize the text from the file and print the number of tokens:')
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    print('\nRemove the first 50 tokens from the encoded text:')
    enc_sample = enc_text[50:]

    print('Number of tokens after removing the first 50 tokens:', len(enc_sample))

    CONTEXT_SIZE = 4
    x = enc_sample[:CONTEXT_SIZE]
    y = enc_sample[1:CONTEXT_SIZE + 1]
    print(f"x: {x}")
    print(f"y:      {y}")

    print('\n----------------------------------------')
    print('Next token prediction task:')

    for i in range(1, CONTEXT_SIZE + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, '---->', desired)

    for i in range(1, CONTEXT_SIZE + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))
