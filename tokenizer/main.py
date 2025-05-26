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

import urllib.request
import re

if __name__ == "__main__":
    print("Downloading 'the-verdict.txt', Edith Wharton’s short story, from GitHub...")

    URL = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    FILE_PATH = "the-verdict.txt"
    urllib.request.urlretrieve(URL, FILE_PATH)

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    print("\nTokenize the text into words and punctuation marks:")
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    print(preprocessed)

    print("\nRemove empty strings from the list:")
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed)

    print("\nTotal number of tokens:", len(preprocessed))

    print("\nFirst 30 tokens:")
    print(preprocessed[:30])

    print("\n----------------------------------------")
    print("Create the sorted vocabulary")
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print("\nVocabulary size:", vocab_size)

    print("\nFirst 50 tokens in the vocabulary:")
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)

        # Limit the output to the first 50 items
        if i >= 50:
            break

    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))
