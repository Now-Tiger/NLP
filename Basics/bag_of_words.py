#!/usr/bin/env tensor 

# --------------------------- Bag of Words ---------------------------

# The name “bag-of-words” comes from the algorithm simply seeking to 
# know the number of times a given word is present within a body of text.

from re import findall
from collections import Counter
from nltk.tokenize import word_tokenize as wt 

sample_text = "'I am a student from the University of Alabama. I\
was born in Ontario, Canada and I am a huge fan of the United\
States. I am going to get a degree in Philosophy to improve\
my chances of becoming a Philosophy professor. I have been\
working towards this goal for 4 years. I am currently enrolled\
in a PhD program. It is very difficult, but I am confident that\
it will be a good decision'"

sample_word_tokenize = wt(sample_text)

print(sample_word_tokenize)

def bow_(text) :
    bag_of_words = [Counter(findall(r'\w+', word)) for word in text]
    bow_ = sum(bag_of_words, Counter())
    return bow_

sample_word_tokenize_bow = bow_(sample_word_tokenize) 
print(sample_word_tokenize_bow)
# sample_word_tokenize_bow2 = bow_(sample_text)

