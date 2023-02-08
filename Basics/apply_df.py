#!/usr/bin/env/ conda:"base"
# -*- coding: utf-8 -*- 

import pandas as pd
import re

from textblob import TextBlob

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

sample ="'I am a student from the University of Alabama.\
 I was born in Ontario, Canada and I am a huge fan of the United States.\
 I am going to get a degree in Philosophy to improve my chances of becoming a Philosophy professor.\
 I have been working towards this goal for 4 years.\
 I am currently enrolled in a PhD program.\
 It's very difficult, but I'm confident that it will be a good decision'" 

if __name__ == "__main__":
    
    re_clean = re.sub(r'[^\w\s]', '', sample.lower())
 
    df = pd.DataFrame(sent_tokenize(sample), columns=['text'])

    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)
    
    # -- now remove stopwords --
    stoppies = stopwords.words('english')

    df['no_stopwords'] = df['text'].apply(lambda x: " ".
            join(x for x in str.split(x.lower()) if x not in stoppies)
        )
    # -- correct spellings --
    df['correct_spells'] = df['no_stopwords'].apply(lambda x: 
            str(TextBlob(x).correct())
        )
    print(df) 

    del df, re_clean, stoppies
