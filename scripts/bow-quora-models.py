#!/usr/bin/env/ python 3.10
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.metrics import (f1_score, recall_score)
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)


def baseline_model(models: dict, inputs: np.ndarray, target: pd.Series, test_inps: np.ndarray, test_target: pd.Series) -> None:
    """ Fits the model from the models dictionary and returns metrics summary from model_metrics function. """
    for name, model in models.items():
        print(f"\n*** {name} *** ")
        model.fit(inputs, target)
        model_metrics(inputs, target, model, 'train set')
        model_metrics(test_inps, test_target, model, 'test set')
    return



def model_metrics(inputs: pd.DataFrame, target: pd.Series, model, name: str) -> str:
    """ returns a performance summary/metrics of the model """
    preds = model.predict(inputs)

    print(f"\n---- {name} ----",
          f"F1 Score: {f1_score(target, preds):.3f}",
          f"Accuracy: {model.score(inputs, target):.3f}",
          f"recall score: {recall_score(target, preds):.3f}", 
          sep="\n"
          )
    return


def tokenizer(text: str) -> list:
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(word) for word in word_tokenize(text)]


def predictions(model, vectorizer, *, sentence: str = None, sentences: np.ndarray = None) -> None:
    if sentence:
        pred = model.predict(vectorizer.transform([sentence]))
        print(f"Input: {sentence}", f"Prediction: {pred}", sep = "\n")
    else:
        for i, sent in enumerate(sentences):
            pred = model.predict(vectorizer.transform(sent.splitlines()))
            print(
                f"sent: {i + 1} | prediction: {'Sincere' if pred == 0 else 'Insincere'}", 
                f"| {pred}", 
                sep='\t'
            )
    return
    


if __name__ == "__main__":
    data = pd.read_csv("../Data/Quora/final_quora_dataset.csv")
    
    vectorizer = CountVectorizer(lowercase=True, tokenizer=tokenizer, max_features=1000,)
    vectorizer.fit(data.clean)
    vec_inputs = vectorizer.transform(data.clean)
    
    train_inpV, val_inpV, train_targetV, val_targetV = split(
        vec_inputs, data.target, test_size=.30, random_state=42, shuffle=True, stratify=data.target)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='saga',),
        "Logistic Reg. CV": LogisticRegressionCV(max_iter=1000, cv=3, random_state=42, solver='saga')
    }

    baseline_model(models, train_inpV, train_targetV, val_inpV, val_targetV)

    # -- TFIDF --
    tfidf = TfidfVectorizer(max_features=1000, tokenizer=tokenizer, lowercase=True, use_idf=True)
    tfidf.fit(data.clean)
    tfidf_inputs = tfidf.transform(data.clean)

    train_idf, val_idf, target_idf, val_target_idf = split(
        tfidf_inputs, data.target, test_size=.30, random_state=42, shuffle=True, stratify=data.target)
    
    baseline_model(models, train_idf, target_idf, val_idf, val_target_idf)


    sent = "I hate all of you people"
    modal = models['Logistic Reg. CV']

    # - dummy data
    input_sentences = np.array([
        "Why can't Indians stop comparing themselves with much developed countries and start working on \
        building themselves instead?",
        "has the united states become the largest dictatorship in the world",
        "I hate this country",
        "I am going to start watching 'the office' webseries",
        "hate this employee so much!", 
        "How to destroy 'LGBT' community?",
        "Can we all now admit that President Trump doesn't really want Congress to pass illegations"], 
        dtype = 'str'
    )

    predictions(sentences=input_sentences, model=modal, vectorizer=tfidf)