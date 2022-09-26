#!/usr/bin/env/ 'conda':base

import pandas as pd
import nltk
from textblob import TextBlob

# ------------- Create data -----------------
text=['Introduction to NLP',
        'It is likely to be useful, to people ',
        'Machine learning is the new electrcity', 
        'R is good langauage',
        'I like this book',
        'I want more books like this']

data = pd.DataFrame({'tweet': text})
#print(data)

data['tweet'].apply(lambda x: str(TextBlob(x).correct()))
print(data)

# :!python correct-spellings.py 
#                                     tweet
# 0                     Introduction to NLP
# 1   It is likely to be useful, to people
# 2  Machine learning is the new electrcity
# 3                     R is good langauage
# 4                        I like this book
# 5             I want more books like this