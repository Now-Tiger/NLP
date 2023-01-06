#!/usr/bin/env 'conda': base 
# -*- coding: utf-8 -*-

import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# There are various ways we can pull out the stem of a word. 
# Here’s a simple-minded approach that just strips off anything that looks like a suffix:
def custom_stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
           return word[:-len(suffix)]
    return word

def port_stemmer(sentence: str) -> None:
    # text = sentence.lower()
    tokenization = word_tokenize(text=sentence)
    for w in tokenization:
        print(f"stemming for {w} is {PorterStemmer().stem(word=w)}")
    return

def lemmatizer(sentence: str) -> None:
    word_lemmatizer = WordNetLemmatizer()
    text = sentence.lower()
    tokenization = word_tokenize(text)
    for word in tokenization:
        print(f"stemming for {word} is {word_lemmatizer.lemmatize(word)}")
    return



if __name__ == "__main__":
    sent = "Studies studying cries cry"
    # port_stemmer(sent)
    # print(custom_stem("previously"))
    # lemmatizer(sent)

    text = ['I like fishing', 'I eat fish', 'There are many fishes in pound', 
            'Leaves and leaf'
           ]
    tweets = pd.DataFrame({'tweets': text})

    # data = tweets['tweets'].apply(lambda x: " ".join([PorterStemmer().stem(word) for word in x.split()]))
    # print(data)