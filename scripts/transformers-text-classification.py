#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import re
from warnings import filterwarnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('seaborn-v0_8')
filterwarnings('ignore')


DATA_DIR = "../Data/Imdb"
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
TEST_DIR = os.path.join(DATA_DIR, 'test/')


SEED = 123
BATCH_SIZE = 1024


class Transformer(keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate=.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation='relu'),
             keras.layers.Dense(embed_dim),]
        )
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)


class TokenAndPostionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(
            input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(
            start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def pack_together() -> tf.keras.Model:
    inputs = keras.layers.Input(shape=(MAX_LENGTH,))
    embedding_layer = TokenAndPostionEmbedding(
        maxlen=MAX_LENGTH, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM)
    x = embedding_layer(inputs)
    transformer_block = Transformer(
        embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    x = transformer_block(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(.1)(x)
    x = keras.layers.Dense(20, activation='relu')(x)
    x = keras.layers.Dropout(.1)(x)
    outputs = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


train_ds = tf.keras.utils.text_dataset_from_directory(
    TRAIN_DIR,
    validation_split=.2,
    batch_size=BATCH_SIZE,
    seed=SEED,
    subset='training',
)

val_ds = tf.keras.utils.text_dataset_from_directory(
    TRAIN_DIR,
    validation_split=.2,
    batch_size=BATCH_SIZE,
    seed=SEED,
    subset='validation'
)

test_ds = tf.keras.utils.text_dataset_from_directory(
    TEST_DIR,
    batch_size=BATCH_SIZE,
    seed=SEED
)


# -- CONFIGURE DATASET FOR PERFORMANCE --
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


train_sentences = []
train_labels = []
val_sentences = []
val_labels = []

for text, label in train_ds.unbatch():
    train_sentences.append(text.numpy())
    train_labels.append(label.numpy())
for ex, l, in val_ds.unbatch():
    val_sentences.append(ex.numpy())
    val_labels.append(l.numpy())

train_label = np.array(train_labels)
val_label = np.array(val_labels)


def custom_processing(input_sentences) -> list:
    table = []
    clean = re.compile(r'<.*?>')
    for sent in input_sentences:
        txt = re.sub(clean, '', sent.decode('UTF-8'))
        t = re.sub(r'[^\w\s]', '', txt.lower())
        table.append(t)
    return table


train_sent = custom_processing(train_sentences)
val_sent = custom_processing(val_sentences)


# -- GENERATE PADED SEQUENCES --
# - params
VOCAB_SIZE = 2000
EMBEDDING_DIM = 32
MAX_LENGTH = 200
NUM_HEADS = 2
FF_DIM = 32

TRUNC_TYPE = 'post'
OOV_TOK = '<OOV>'


# -- INITIALIZE TOKENIZER --
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(train_sent)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sent)
train_paded = tf.keras.utils.pad_sequences(
    train_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

val_sequences = tokenizer.texts_to_sequences(val_sent)
val_paded = tf.keras.utils.pad_sequences(
    val_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)


model = pack_together()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_paded,
    train_label,
    batch_size=32,
    epochs=2,
    validation_data=(val_paded, val_label)
)

hist = pd.DataFrame(history.history)

def plot() -> None:
    # -- PLOT LURNING CURVES: ACCURACY & LOSS --
    plt.figure(figsize=(10, 8), dpi=90)
    plt.subplot(211)
    plt.title("Cross-Entropy loss", pad=10)
    plt.plot(hist['loss'], label='training', color='skyblue')
    plt.plot(hist['val_loss'], label='validation', color='teal')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8), dpi=90)
    plt.subplot(212)
    plt.title('Accuracy', pad=10)
    plt.plot(hist['accuracy'], label='training', color='skyblue')
    plt.plot(hist['val_accuracy'], label='validation', color='teal')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


plot()
