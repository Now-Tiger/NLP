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
from nltk.corpus import wordnet as wn
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
welcome_response = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


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


with open(PATH + FILENAME, 'r') as file:
    text = " ".join(x.strip().lower() for x in file.readlines())


def process_text(text) -> None:
    remove_punct_dict = dict((ord(punct), None)for punct in string.punctuation)
    word_token = word_tokenize(text.lower().translate(remove_punct_dict))
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')
        new_words.append(new_word)
    # - Remove tags
    rmv = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        rmv.append(text)
    # - pos tagging and lemmatization
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list


def generate_response(user_response: str = None) -> None:
    robo_response = ''
    # * Fix this problem sent_tokens not initialized,
    # reason of error: we moved code from if __name__ == "__main__" under the main method
    # FIX SOON
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=process_text, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0) or "tell me about" in user_response:
        print("Checking Wikipedia")
        if user_response:
            robo_response = wikipedia_data(user_response)
            return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


def wikipedia_data(input) -> str:
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            wiki = wk.summary(topic, sentences=3)
            return wiki
    except Exception as e:
        print("No content has been found")


def main() -> None:
    FLAG: bool = True
    print("This is wiki chatbot. Start typing to ask questions.\nTo exit enter bye")
    while FLAG:
        user_response = str(input("\n>> "))
        user_response = user_response.lower()
        if(user_response not in ['bye', 'shutdown', 'exit', 'quit']):
            if(user_response == 'thanks' or user_response == 'thank you'):
                FLAG = False
                print(">> Bot : You are welcome..")
            else:
                if(welcome(user_response) != None):
                    print(f">> Bot : {welcome(user_response)}")
                else:
                    print(">> Bot : ", end="")
                    print(generate_response(user_response))
                    sent_tokens.remove(user_response)
        else:
            FLAG = False
            print(">> Bye! ")


if __name__ == "__main__":
    maybe_download()
    with open(PATH + FILENAME, 'r') as file:
        raw = " ".join(x.strip().lower() for x in file.readlines())
    sent_tokens = sent_tokenize(raw)
    main()