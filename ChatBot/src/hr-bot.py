#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8-
import os
import re
import string
import random
import unicodedata
from urllib.request import urlopen
from collections import defaultdict

import wikipedia as wk
from nltk import pos_tag
from bs4 import BeautifulSoup as bs
from nltk.corpus import (wordnet, stopwords)
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import (word_tokenize, sent_tokenize)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import (cosine_similarity, linear_kernel)

from warnings import filterwarnings
filterwarnings("ignore")


PATH: str = "../Data/"
FILENAME: str = 'HR.txt'
URL: str = "https://www.whatishumanresource.com/human-resource-management"


welcome_input = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
welcome_response = ["hi", "hey", "*nods*", "hi there",
                    "hello", "I am glad! You are talking to me"]


def welcome(user_response):
    for word in user_response.split():
        if word.lower() in welcome_input:
            return random.choice(welcome_response)


def maybe_download() -> None:
    response = urlopen(URL)
    html_doc = response.read()
    soup_ = bs(html_doc, 'html.parser')
    with open(PATH + FILENAME, 'w', encoding='utf-8') as file:
        for x in soup_.find_all('p'):
            file.writelines(x.text.lower().strip())
    return


def save_tokens(path: str, tokens: list, type_of_tokens: str) -> None:
    if not os.path.exists(path):
        print(f"'{path}' does not exist")
    else:
        with open(path + type_of_tokens, 'w') as file:
            for token in tokens:
                file.writelines(token)
                file.writelines('\n')
    return


def process_text() -> None:
    new_words = []
    with open(PATH + FILENAME, 'r') as file:
        raw = " ".join(x.strip().lower() for x in file.readlines())
    punctuation_dict = dict((ord(punct), None) for punct in string.punctuation)
    word_tokens = word_tokenize(raw.translate(punctuation_dict))
    for word in word_tokens:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    rmv = []
    for w in new_words:
        text = re.sub('&lt;/?.*?&gt;', '&lt;&gt;', w)
        text = re.sub(r'[0-9]', '', w)
        rmv.append(text)
    stoppies = stopwords.words('english')
    rmvd = [word for word in rmv if word not in stoppies]
    tag_map = defaultdict(lambda: wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in pos_tag(rmvd):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list


def generate_response(user_response: str = None) -> None:
    with open(PATH + FILENAME, 'r') as file:
        raw = " ".join(x.strip().lower() for x in file.readlines())
    bot_resp = ''
    sent_tokens = sent_tokenize(raw)
    sent_tokens.append(user_response)
    tfidf_vec = TfidfVectorizer(tokenizer=process_text, stop_words='english')
    tfidf = tfidf_vec.fit_transform(sent_tokens)
    values = linear_kernel(tfidf[-1], tfidf)
    indexes = values.argsort()[0][-2]
    flat = values.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0 or "tell me about" in user_response:
        print("checking wikipedia...")
        if user_response:
            bot_resp = wikipedia_data(user_response)
            return bot_resp
        else:
            bot_resp += sent_tokens[indexes]
            return bot_resp


def wikipedia_data(input) -> str:
    regex = re.search('tell me about (.*)', input)
    try:
        if regex:
            topic = regex.group(1)
            wiki = wk.summary(topic, sentences=3)
            return wiki
    except Exception as err:
        print(f"No content has been found: \n{err}")


if __name__ == "__main__":
    lemma = process_text()  # <- working properly
    generate_response()