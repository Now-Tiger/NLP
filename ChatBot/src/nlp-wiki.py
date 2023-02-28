#!/usr/bin/env/ conda: "base"
# -*- coding: utf-8 -*-
import re
import string
import random

from urllib import request

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import (WordNetLemmatizer)
from nltk.tokenize import (word_tokenize, sent_tokenize)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

""" Doesn't work properly... Fix bugs"""


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


def lem_tokens(tokens: list) -> list:
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]


def lem_normalize(text: str) -> list:
    return lem_tokens(word_tokenize(text.lower().translate(remove_punc_dict)))


if __name__ == "__main__":
    URL = "https://en.wikipedia.org/wiki/Natural_language_processing"
    response = request.urlopen(URL)
    html_doc = response.read()
    soup = BeautifulSoup(html_doc, 'html.parser')

    corpus_list = []

    """ str.strip() removes '\n' for the text"""
    for x in soup.find_all('p'):
        corpus_list.append(x.text.strip())

    corpa = " ".join(x.strip() for x in corpus_list)
    corpa = re.sub(r'[^\w\s]', '', corpa.lower())
    corpa = re.sub(r'[0-9]', '', corpa.lower())

    word_tokens = word_tokenize(corpa)
    sent_tokens = sent_tokenize(corpa)

    stoppies = stopwords.words('english')
    non_stops = [word for word in word_tokens if word not in stoppies]

    for word in non_stops:
        if word == "e.g" or word == "eg":
            non_stops.remove(word)

    # ------------------------------
    greet_inputs = ('hello', 'hi', 'whats up', 'how are you?')
    greet_responses = ('hi! ', 'hello. ', 'Hey There!!', 'There there!')
        
    def greet(sentence: str) -> str:
        for word in sentence.split():
            if word.lower() in greet_inputs:
                return random.choice(greet_responses)
    
    def response(user_response: str) -> str:
        bot_response = ''
        TFIDF_vec = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
        tfidf = TFIDF_vec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf == 0:
            bot_response += "I'm sorry. Unable to understand you."
            return bot_response
        else:
            bot_response += sent_tokens[idx]
            return bot_response

    flag = True
    print(f"Hello this is Learning bot !",
          f"\nStart typing your text after greetings to talk.", 
          f"\nTo exit type bye!"
        )
    while flag:
        user_response = input()
        user_response = user_response.lower()
        if (user_response != 'bye!'):
            if (user_response == 'thank you') or (user_response == 'thanks'):
                flag = False
                print('Bot: You are welcome')
            else:
                if (greet(user_response) != None):   # <-- fix this line of code
                    print(greet(user_response))
                else:
                    sent_tokens.append(user_response)
                    word_tokens = word_tokens + word_tokenize(user_response)
                    final_words = list(set(word_tokens))
                    print('Bot: ', end = '')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print('Bot: Goodbye!')