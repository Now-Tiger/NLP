#!/usr/bin/env/ python 3.10
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import (f1_score, recall_score)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (LogisticRegressionCV, LogisticRegression)

from warnings import filterwarnings
filterwarnings('ignore')


def baseline_model(models: dict, inputs: np.ndarray, target: pd.Series, test_inps: np.ndarray, test_target: pd.Series) -> None:
    """ Fits the model from the models dictionary and returns metrics summary from model_metrics function. """
    for name, model in models.items():
        print(f"\n*** {name} *** ")
        model.fit(inputs, target)
        model_metrics(inputs, target, model, 'train set')
        model_metrics(test_inps, test_target, model, 'test set')
    return


def model_metrics(inputs: np.ndarray, target: pd.Series, model, name: str) -> str:
    """ Returns a performance summary/metrics of the model """
    preds = model.predict(inputs)

    print(f"\n---- {name} ----",
          f"F1 Score: {f1_score(target, preds):.3f}",
          f"Accuracy: {model.score(inputs, target):.3f}",
          f"recall score: {recall_score(target, preds):.3f}",
          sep="\n"
          )
    return


def read_data(filename: str, sep: str = None, names: list | str = None) -> pd.DataFrame:
    """ Read the csv dataset and returns data in Pandas DataFrame object. """
    data = pd.read_csv(filename, sep = sep, names = names)
    return data


def process_text(text: str) -> str:
    """ BUGS Here, Fix required """
    clean = re.compile(r'<.*?>')
    for sent in text.split():
        tex = re.sub(clean, '', sent.lower())
        te = re.sub(r'[\w\s]', '', tex)
        t = re.sub(r'[0-9]', '', te)
    return te


def tokenizer(text: str) -> list:
    """ Word tokenizer """
    stemmer = SnowballStemmer(language='english')
    return [stemmer.stem(word) for word in word_tokenize(text)]


def vectorization(max_features: int, df: pd.DataFrame, column: str) -> np.ndarray:
    """ TfIdfVectorization applies on input text corpus and returns the vectorized text. """
    vectorizer = TfidfVectorizer(
        lowercase = True,
        tokenizer = tokenizer,
        max_features = max_features,
        use_idf = True
    )
    vectorizer.fit(df[column])
    inputs = vectorizer.transform(df[column])
    return inputs


def split_data(vectorized_inputs: np.ndarray, target: pd.Series, test_size: int) -> np.ndarray:
    """ Splits and returns the dataset input and target in 4 pieces to train on model """
    train_inp, val_inp, train_target, val_target = split(
        vectorized_inputs,
        target,
        test_size = test_size,
        random_state = 42,
        stratify = target
    )
    return (train_inp, val_inp, train_target, val_target)


def main() -> None:
    PATH = '../Data/SMSspams/'
    FILE = 'SMSSpamCollection'
    df = read_data(
        Path(PATH + FILE),
        sep='\t',
        names=['label', 'message']
    )

    stoppies = stopwords.words('english')

    # MAKE A FUNCTION THAT DOES THE PROCESSING RATHER THAN MANUALE PROCESSING
    df['clean'] = (
        df['message'].str.replace(r'[0-9]', '', regex=True)
        .str.replace(r'[#£^.*<>?!+=/)(%]', '', regex=True)
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'http', '', regex=True)
        .str.replace(r'com', '', regex=True)
        .str.replace(r'www', '', regex=True)
        .apply(lambda x: " ".join(x.lower() for x in x.split() if x not in stoppies))
    )

    target = df['label'].map({'ham': 0, 'spam': 1})
    inputs = vectorization(max_features=1000, df=df, column='clean')
    train_inp, val_inp, train_target, val_target = split_data(inputs, target, 0.30)

    # ** Model table **
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='saga',),
        "Logistic Reg. CV": LogisticRegressionCV(max_iter=1000, cv=3, random_state=42, solver='saga')
    }
    baseline_model(models, train_inp, train_target, val_inp, val_target)
    return


if __name__ == "__main__":
    main()
