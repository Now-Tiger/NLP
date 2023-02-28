#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-
import os
import re
import string
from urllib import request

from bs4 import BeautifulSoup as soup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import (word_tokenize, sent_tokenize)
from nltk.corpus import stopwords

URL: str = "https://en.wikipedia.org/wiki/Natural_language_processing"
PATH: str = "../Data/"
FILENAME: str = "nlp_new_wiki.txt"


lemmer = WordNetLemmatizer()
punctuations_dictionary = dict((ord(punct), None) for punct in string.punctuation)


def maybe_download() -> None:
    response = request.urlopen(URL)
    html_doc = response.read()
    soup_ = soup(html_doc, 'html.parser')

    with open(PATH + FILENAME, 'w', encoding='utf-8') as file:
        for x in soup_.find(id='bodyContent').find_all('div'):
            file.writelines(x.text.strip())
    return


def read_and_tokenize_wiki() -> list:
    with open(PATH + FILENAME, 'r', errors='ignore') as file:
        raw = " ".join(x.strip().lower() for x in file.readlines())
    sent_tokens = sent_tokenize(raw)
    word_tokens = word_tokenize(raw)
    return (sent_tokens, word_tokens)


def lemmatize(tokens) -> list:
    return [lemmer.lemmatize(token) for token in tokens]


def lem_normalize(text: str) -> list:
    return lemmatize(word_tokenize(text.lower().translate()))


if __name__ == "__main__":
    maybe_download()
    sent_tokens, word_tokens = read_and_tokenize_wiki()
    lemmatize(word_tokens)