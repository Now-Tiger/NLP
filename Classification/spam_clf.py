#!/usr/bin/env/ conda: "tensor"
# -*- coding: utf-8 -*-
import os
import re

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as split

from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def custom_processing(input_sentences) -> list:
    """ Method cleans the input text data 
        removes unwanted html tags, punctuations

    Args:
    ----
        input_sentences (np.ndarray): _description_

    Returns:
    ----
        list: list containing cleaned/processed data.
    """
    table = []
    clean = re.compile(r'<.*?>')
    for sent in input_sentences:
        txt = re.sub(clean, '', sent)
        t = re.sub(r'[^\w\s]', '', txt.lower())
        table.append(t)
    return table


def fit_tokenizer(
        input_text: np.ndarray,
        vocab_size: int,
        oov_token: str,
        max_length: int,
        trucation_type: str) -> np.ndarray:
    """ Method fits the tensorflow tokenizer on the given 
    text input present inside numpy arrays or lists.

    Args:
    ----
        input_text (np.ndarray): numpy array/list carrying text data.

    Returns:
    ----
        np.ndarray: return numpy array as padded sequences of the
                    fitted text data. 
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(input_text)
    sequences = tokenizer.texts_to_sequences(input_text)
    paded = tf.keras.utils.pad_sequences(
        sequences, maxlen=max_length, truncating=trucation_type)
    return paded


def model(
    vocab_size: int,
    embedding_dim: int,
    input_len: int,
    paded: np.ndarray,
    target: np.ndarray,
    val_paded: np.ndarray,
    val_target: np.ndarray,
    epochs: int 
    ) -> pd.DataFrame:
    """ Fitting sequential model with embeddings returns 
    a pandas dataframe object storing model's accuracy and loss

    Args:
    ----
        vocab_size (int): total vocabulary size
        embedding_dim (int): embeddings dimensions
        input_len (int): input lenth for the embeddings
        paded (np.ndarray): paded training matrix of text
        target (np.ndarray): array containing true labels
        val_paded (np.ndarray): paded validation matrix 
        val_target (np.ndarray): true labels of validation
        epochs (int): number of epochs to iterate model that number of times.

    Returns:
    ----
        pd.DataFrame: Storing model results on a dataframe.
    """
    net = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=input_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    net.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    result = net.fit(paded, target, validation_data=(
        val_paded, val_target), epochs=epochs)
    df = pd.DataFrame(result.history)
    return df


def plot_result(dataframe: pd.DataFrame) -> plt.plot:
    # - plot loss curves (train & validation)
    plt.figure(figsize=(10, 8), dpi=90)
    plt.subplot(211)
    plt.title("Cross-Entropy loss", pad=10)
    plt.plot(dataframe['loss'], label='training', color='skyblue')
    plt.plot(dataframe['val_loss'], label='validation', color='teal')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    # plot accuracy learning curves
    plt.figure(figsize=(10, 8), dpi=90)
    plt.subplot(212)
    plt.title('Accuracy', pad=10)
    plt.plot(dataframe['accuracy'], label='training', color='skyblue')
    plt.plot(dataframe['val_accuracy'], label='validation', color='teal')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def main() -> None:
    data = pd.read_csv('../data/ham-spam.csv')
    text = np.array(data.Text.values)
    target = np.array(data.IsSpam.values)

    train_inp, val_inp, train_target, val_target = split(
        text, target, test_size=.25, random_state=42, shuffle=True)

    train_cleaned = custom_processing(train_inp)
    val_cleaned = custom_processing(val_inp)

    # - params
    vocab_size: int = 10_000
    max_length: int = 120
    embedding_dim: int = 16
    truncation_type: str = 'post'
    oov_token = '<OOV>'
    epochs: int = 10

    train_paded = fit_tokenizer(
        train_cleaned, vocab_size, oov_token, max_length, truncation_type)
    val_paded = fit_tokenizer(
        val_cleaned, vocab_size, oov_token, max_length, truncation_type)

    res = model(
        vocab_size,
        embedding_dim,
        max_length,
        train_paded,
        train_target,
        val_paded,
        val_target,
        epochs,
    )
    plot_result(res)


if __name__ == "__main__":
    # * Model accuracy is low and errors are hight
    # * Fine tune preprocessing, remove stop words before hand
    # * Fine tune model architecture
    main()
