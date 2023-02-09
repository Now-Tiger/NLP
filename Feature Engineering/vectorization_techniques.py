#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer, TfidfVectorizer)

text = ["The quick brown fox jumped over the lazy dog."]

sample = "I am a studant from the University of Alabama.\
 I was born in Ontario, Canada and I am a huge fan of the United States.\
 I am going to get a degree in Philosophy to improv my chances of becoming a Phylosopi professer.\
 I have been working towards this goal for 4 years.\
 I am currently enrolled in a PhD program.\
 It's very dificult, but I'm confident that it will be a good decision"


if __name__ == "__main__":
    # -- Implement Count Vectorization before Hash Vectorization below --
    sentences = sent_tokenize(sample)
    df = pd.DataFrame(sentences, columns=['text'])

    vectorizer = HashingVectorizer(
        n_features=20, decode_error="ignore", alternate_sign=False).fit(df.text)
    vec = vectorizer.transform(df.text)

    print(vec.shape, vec.toarray(), vectorizer.get_stop_words(), sep="\n\n")

    del (df, vectorizer, vec, sentences, sample, text)
