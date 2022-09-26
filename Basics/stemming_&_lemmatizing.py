#!/usr/bin/env 'conda': base 

import pandas as pd
from textblob import Word, TextBlob
from nltk.stem import PorterStemmer

text = ['I  like fishing', 'I eat fish', 'There are many fishes in pound']
data = pd.DataFrame({'tweet': text})

# -------------------------------- Stemming the text -----------------------------------

st = PorterStemmer()
stemmed = data['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
print(stemmed)


# -------------------------------- Lemmatizing the text --------------------------------

"""
        Lemmatization is a process of extracting a root word by considering the vocabulary. 
    For example: "good","better", or "best" is lemmatized into good.

        The part of speech of a word is determined in lemmatization. It will
    return the dictionary form of a word, which must be a valid word while
    stemming just extracts the root word.
"""

new_text = ['I like fishing', 'I eat fish', 'There are many fishes in pound', 
            'Leaves and leaf']

new_data = pd.DataFrame({'tweet': new_text})

new_data = new_data['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(new_data['tweet'])

