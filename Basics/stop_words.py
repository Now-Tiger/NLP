#!/usr/bin/env 'base':conda


# Problem you want to remove stop words:
#   - Simplest way is using NLTK library.

# import libraries first

from nltk.corpus import stopwords 
import pandas as pd
import re

from warnings import filterwarnings
filterwarnings("ignore")


# ---------- creat text data -----------

text = ['This is introduction to NLP','It is likely to be useful, to people ',
        'Machine learning is the new electrcity',
        'There would be less hype around AI and more action going forward',
        'python is the best tool!','R is good langauage',
        'I like this book',
        'I want more books like this']

data = pd.DataFrame({"tweet":text})

# print(data)

stop = stopwords.words('english')
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data['tweet'])