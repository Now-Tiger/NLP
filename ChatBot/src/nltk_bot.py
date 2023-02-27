#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-
import os
import re
import string
from urllib import request

from bs4 import BeautifulSoup as soup
from nltk.tokenize import (word_tokenize, sent_tokenize)
from nltk.corpus import stopwords


URL: str = "https://en.wikipedia.org/wiki/Natural_language_processing"
PATH: str = "../Data/"
FILENAME: str = "nlp_wiki.txt"


def maybe_download() -> None:
    response = request.urlopen(URL)
    html_doc = response.read()
    soup_ = soup(html_doc, 'html.parser')
    if FILENAME not in os.listdir(PATH):
        print(f"downloading file.. '{FILENAME}'")
    with open(PATH + FILENAME, 'w') as file:
        for x in soup_.find_all('p'):
            file.writelines(x.text)
    return


def clean_corpus() -> list:
    with open(PATH + FILENAME, 'r') as lines:
        corpus = " ".join(
            x.strip() for x in lines.readlines()
        )
    stoppies = stopwords.words('english')
    sent_tokens = sent_tokenize(str.lower(corpus))
    word_tokens = word_tokenize(str.lower(corpus))
    cleans = [re.sub(r'[^\w\s\0-9]', '', token) for token in word_tokens]
    without_stops = [
        word for word in cleans if word not in stoppies
    ]
    print(without_stops)


if __name__ == "__main__":
    clean_corpus()
    """ try different regex sub r'\([^)]*\)' 
        Because many punctuations are staying in the corpus as it is.
    """