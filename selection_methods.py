"""
David Rubio Vallejo
Ling131 Final project

Spanish chatbox V.1

This module contains the implementation of the selection algorithms:
the basic one (random), the one based on the Jaccard coefficient, and the one based on the TD-IDF measure.
"""

import os
import math
import nltk
import pickle
import random
import collections
from preprocessing import make_vocabulary_list, make_list_utterances

# Loads and stores the dictionary with the vocabulary words (keys) and the sentences where they appear (values)
file1 = open("Freq_dict.pkl", "rb")
freq_dict = pickle.load(file1)


def basic_selection(tokenized_input):
    """For each word in the tokenized input, get the sentences from the training data where it
    appears and select one of them at random """
    answer_list = []
    for word in tokenized_input:
        answer_list.extend(freq_dict.get(word, ["no sé qué es " + word]))
    return random.choice(answer_list)


def jaccard_selection(tokenized_input):
    """The sentence selected is one that maximizes the Jaccard coefficient between itself and the input sentence."""

    # Collects all the sentences in which the words in the input appear. If an input word does not appear in the
    # dict, the Spanish equivalent of "I don't know what [word] is", will be added
    sent_list = []
    for word in tokenized_input:
        sent_list.extend(freq_dict.get(word, ["no sé qué es " + word]))

    jacc_coef_dict = collections.defaultdict(list)

    # For each sentence stored above...
    for sent in sent_list:
        # Tokenize it and remove punctuation
        tokd_sent = nltk.word_tokenize(sent.lower(), language="spanish")
        clean_tokd_sent = [x for x in tokd_sent if x.isalpha()]
        # Create a list with the contents of the input and the sent at stake (union)
        union_list = clean_tokd_sent + tokenized_input
        union_list.sort()
        # Create a list with the elements that appear both in the input and in the sent at stake (intersection)
        intersection_list = []
        for index in range(0, len(union_list) - 1):
            if union_list[index] == union_list[index + 1]:
                intersection_list.append(union_list[index])
        # Size union = |A| + |B| - |A int B| (doing a set eliminates any duplicates that may result for two words
        # appearing together
        size_union = len(list(set(union_list)))
        size_intersection = len(list(set(intersection_list)))
        jacc_coef = size_intersection / size_union
        # Add to the dict of coeficients the coeficient (if it wasn't there already) and the sent at stake
        jacc_coef_dict[jacc_coef].append(sent)

    # Returns a random sent from among the ones with the highest similarity to the original one
    return random.choice(jacc_coef_dict[max(jacc_coef_dict.keys())])


def tf_idf_selection(tokenized_input, idf_dct):
    """The sentence selected is one that contains the word from the input sentence that maximizes the TF-IDF
    coefficient between itself and the selected sentence"""

    # Collects all the sentences in which the words in the input appear. If an input word does not appear in the
    # dict, the Spanish equivalent of "I don't know what [word] is", will be added
    sent_list = []
    for word in tokenized_input:
        sent_list.extend(freq_dict.get(word, ["no sé qué es " + word]))

    # This dict stores the tf-idf scores of each word and the list of sentences that lead to said score
    tf_idf_dict = collections.defaultdict(list)

    # For each word in the input sentence and each sentence 'sent'...
    for word in tokenized_input:
        for sent in sent_list:
            tokd_sent = nltk.word_tokenize(sent.lower(), language="spanish")
            clean_tokd_sent = [x for x in tokd_sent if x.isalpha()]

            # Calculates the frequency of that word in the given sentence
            counter = collections.Counter(clean_tokd_sent)
            tf = counter[word] / len(clean_tokd_sent)
            # Gets the IDF score of "word". If it's not in the training set, return 0
            idf = idf_dct.get(word, 0)
            tf_idf = tf * idf

            tf_idf_dict[(tf_idf, word)].append(sent)

    # If no words where added to the tf-idf dict because the input was all stopwords for example, 'max' would throw
    # an exception
    if len(tf_idf_dict) > 0:
        highest_tfidf_tuple = max([x for x in tf_idf_dict.keys()])
        # Option1: Get a random sentence from among the ones that maximize the tf-idf score.
        return random.choice(tf_idf_dict.get(highest_tfidf_tuple))

        # Option2: Get a random sentence that contains the word that maximizes the tf-idf score. This makes the
        # tf-idf score to have less importance because you might end up selecting a sentence where the word does
        # appear, but which otherwise would result in a rather low tf-idf score. The reason that one might wanna do
        # this could be to make the output of the CPU less predictable
        # return random.choice(freq_dict[highest_tfidf_tuple[1]])

    else:
        return "no sé nada de eso"


def idf_calculator(vocabulary, utterance_list):
    """Calculates the IDF of the words in the training data (will take a little while: the vocab is 26156 words long)"""
    output_file = open("IDF_dict.pkl", "wb")
    num_of_utterances = len(utterance_list)
    idf_dct = {}

    for word in vocabulary:
        # Adds one to smooth the result just in case the tokenizer misses some punctuation
        count = 1
        for sent in utterance_list:
            if word in sent:
                count = count + 1
        word_idf = math.log10(num_of_utterances / count)
        idf_dct[word] = word_idf

    pickle.dump(idf_dct, output_file, -1)
    output_file.close()


# Uncomment to create a new pickling file that will hold the dictionary of words and their IDF coefficient
#if __name__ == "__main__":

    #vocab = make_vocabulary_list(os.path.join(os.getcwd(), 'corpus_files\\AAMegaDoc.txt'))
    #tok_utterance_list = [nltk.word_tokenize(sent.lower(), language='spanish') for sent in
    #                     make_list_utterances(os.path.join(os.getcwd(), 'corpus_files\\AAMegaDoc.txt'))]

    #idf_calculator(vocab, tok_utterance_list)
