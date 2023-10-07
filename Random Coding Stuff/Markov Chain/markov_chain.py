import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

#Function to make pairs from an array of text
def make_pairs(text):
    for i in range(0, len(text)-1):
        yield (text[i],text[i+1])

#Extract string from a txt file and call make_pairs function to create word pairs
with open("input.txt", "r") as fp:
    text = fp.read()
corpus = text.split()
pairs = make_pairs(text)

#Populate a word dictionary for the pairs of words
word_dict = dict()
for word_1, word_2 in pairs:
    if word_1 in word_dict.keys():
        word_dict[word_1].append(word_2)
    else:
        word_dict[word_1] = word_2

#Select a first word, if its a lower case, then keep repeating until it is a capital letter
first_word = np.random.choice(text)
while first_word.islower():
    first_word = np.random.choice(text)

#Create a Markov Chain using the first word previosly obtained
chain = [first_word]
n_words = 30
for i in range(n_words):
    chain.append(np.random.choice(word_dict[chain[-1]]))
" ".join(chain)



