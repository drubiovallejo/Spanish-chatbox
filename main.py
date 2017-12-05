"""
David Rubio Vallejo
Ling131 Final project

Spanish chatbox V.1

This is the main module, which is to be called directly for the chatbox to run.
It contains the interface with the user and calls either of the selection algorithms:
the basic one (random), the one based on the Jaccard coefficient, and the one based on the TD-IDF measure.
"""

import pickle
import nltk
from nltk.corpus import stopwords
from selection_methods import basic_selection, jaccard_selection, tf_idf_selection

# Loads and stores the dictionary with the vocabulary words (keys) and their IDF value
file = open("IDF_dict.pkl", "rb")
idf_dict = pickle.load(file)

print("¡Hola! ¿De qué quieres hablar hoy?")
user_input = input()

while user_input != "quit":
    # Tokenizes the input and cleans it by removing punctuation and stopwords
    tokd_input = nltk.word_tokenize(user_input.lower(), language="spanish")
    clean_tokd_input = [x for x in tokd_input if x.isalpha() and x not in stopwords.words("spanish")]

    # Uncomment to use the basic selection algorithm (random)
    # print(basic_selection(clean_tokd_input))

    # Uncomment to use the selection algorithm based on the Jaccard coefficient
    # print(jaccard_selection(clean_tokd_input))

    # Uncomment to use the selection algorithm based on the TF-IDF coefficient
    print(tf_idf_selection(clean_tokd_input, idf_dict))

    user_input = input()

# Goodbye message when user quits
print("¡Hasta luego!")
