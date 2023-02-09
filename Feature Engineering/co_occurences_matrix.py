#!/usr/bin/env/ conda:"base"
# -*- coding: utf-8 -*-
import itertools

import pandas as pd
import numpy as np

import nltk
from nltk import bigrams


def co_occurences_matrix(corpus: list):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_to_index = {word: i for i, word in enumerate(vocab)}
    bi_grams = list(bigrams(corpus))

    # -- frequency distribution of bigrams --
    bigrams_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    co_occurece_mat = np.zeros((len(vocab), len(vocab)))

    # -- loop through the bigrams taking the current and previous word.
    # and number of occurences of the bigram
    for bigram in bigrams_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_to_index[current]
        pos_prev = vocab_to_index[previous]
        co_occurece_mat[pos_current][pos_prev] = count

    co_occurece_mat = np.matrix(co_occurece_mat)
    return co_occurece_mat, vocab_to_index


if __name__ == "__main__":
    sentences = [['I', 'love', 'nlp'], ['I', 'love', 'to' 'learn'],
                 ['nlp', 'is', 'future'], ['nlp', 'is', 'cool']
                 ]
    merged = list(itertools.chain.from_iterable(sentences))
    matrix, vocab_to_index = co_occurences_matrix(merged)

    CoMatrixFinal = pd.DataFrame(
        matrix, index=vocab_to_index, columns=vocab_to_index)
    print(CoMatrixFinal)
