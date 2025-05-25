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
URL = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
FILE_PATH = "the-verdict.txt"
urllib.request.urlretrieve(URL, FILE_PATH)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
