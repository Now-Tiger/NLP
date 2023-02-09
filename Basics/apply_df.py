#!/usr/bin/env/ conda:"base"
# -*- coding: utf-8 -*-

from collections import namedtuple

import pandas as pd

from textblob import TextBlob, Word

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


words = namedtuple("correct_spells", [
                   "student", "Philosophy", "improve", "professor", "difficult"]
                  )

sample = "I am a studant from the University of Alabama.\
 I was born in Ontario, Canada and I am a huge fan of the United States.\
 I am going to get a degree in Philosophy to improv my chances of becoming a Phylosopi professer.\
 I have been working towards this goal for 4 years.\
 I am currently enrolled in a PhD program.\
 It's very dificult, but I'm confident that it will be a good decision"


if __name__ == "__main__":
    df = pd.DataFrame(sent_tokenize(sample), columns=['text'])
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)

    # -- now remove stopwords --
    stoppies = stopwords.words('english')
    df['no_stopwords'] = df['text'].apply(lambda x: " ".join(
        x for x in str.split(x.lower()) if x not in stoppies))

    # -- correct spellings --
    df['correct_spells'] = df['no_stopwords'].apply(
        lambda x: str(TextBlob(x).correct()))
    # -- No change: if we keep first occurance of philosophy word as Phylosphi, we get incorrect words.
    # -- corrections: studant, dificult, improv, professer got corrected.

    # -- Lemmatization --
    df['lemmatized'] = df['correct_spells'].apply(
        lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    del df, stoppies
