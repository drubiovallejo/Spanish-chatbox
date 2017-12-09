"""
David Rubio Vallejo
Ling131 Final project

Spanish chatbox V.1

This module contains the methods that extract the data from the corpus files used for training.
"""

import re
import os
import sys
import nltk
import collections
import pickle
import random
from nltk.tokenize import sent_tokenize, word_tokenize


def read_file_as_string(file_path):
    """Reads file into string"""
    file = open(file_path, 'r')
    string = file.read()
    file.close()
    return string


def read_file_as_list(file_path):
    """Reads file into a list"""
    file = open(file_path, 'r')
    lst = file.readlines()
    file.close()
    return lst


def combine_files(directory):
    """Takes all the .txt files in the given directory and combines them into a single file"""
    dir_path = os.path.join(os.getcwd(), directory)
    megadoc = open(os.path.join(dir_path, "AAMegaDoc.txt"), 'w')
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            st_content = read_file_as_string(os.path.join(dir_path, file_name))
            megadoc.write(st_content)
            megadoc.write("\n")
    megadoc.close()


def clean_file(file_string):
    """Eliminates tags like "laughter", "foreign term" etc. from a string from the corpus data"""
    regex = re.compile(r"\[.*?\]")
    string = regex.sub('', file_string)
    return string


def file_info(file_name):
    """Extracts information lines from file (all such lines begin with '#')"""
    st = read_file_as_string(file_name)
    regex = re.compile(r"#.*")
    lst = regex.findall(st)
    return lst


def make_dict_utterances(file_path):
    """Makes a dictionary of (speaker, utterance) tuples and pickles it"""
    output = open('UtteranceDict.pkl', 'wb')
    dct = collections.defaultdict(list)
    file_list = read_file_as_list(file_path)
    for elem in file_list:
        # Each utterance begins with 'H' (for 'hablante' (Spanish for 'speaker'))
        if elem[:1] == "H":
            # Append starting at 3 to avoid the ":" after each "H" and strip to remove spaces and \n at start and end
            dct[elem[:2]].append(elem[3:].strip())
    pickle.dump(dct, output, -1)
    output.close()


def make_list_utterances_pkl(file_path):
    """Makes a list of all utterances and pickles it"""
    output = open('UtteranceList.pkl', 'wb')
    lst = []
    file_list = read_file_as_list(file_path)
    for elem in file_list:
        if elem[:1] == "H":
            lst.append(elem[3:].strip())
    pickle.dump(lst, output, -1)
    output.close()


def make_vocabulary_list(file_path):
    """Makes a list with the vocabulary (single occurrences of each word)"""
    st = read_file_as_string(file_path)
    normalized_list = set([x for x in nltk.word_tokenize(st.lower(), language='spanish')])
    word_list = list([x for x in normalized_list if x.isalpha()])
    return word_list


def make_list_utterances(file_path):
    """Makes a list of all utterances shorter than 60 (arbitrary length) and returns it"""
    lst = []
    file_list = read_file_as_list(file_path)
    for elem in file_list:
        if elem[:1] == "H" and len(elem) < 60:
            lst.append(elem[3:].strip())
    return lst


def make_freq_dict(vocab, tok_utterance_list):
    """Creates a dictionary the words from the vocabulary as keys and a list of sentences where that word appears as
    values (will take a little while: the vocab is 26156 words long)"""
    output = open("Freq_dict.pkl", "wb")
    dct = collections.defaultdict(list)
    word = 0
    for vocab_word in vocab:
        for sent in tok_utterance_list:
            if vocab_word in sent:
                dct[vocab_word].append(' '.join(sent))
        word = word + 1
        #print(word)
    pickle.dump(dct, output, -1)
    output.close()


def dict_counter(file_path):
    """Given a file, it creates a counter object of the tokenized elements in the file"""
    file = open(file_path, 'r')
    st = file.read()
    lst = nltk.word_tokenize(st, language='spanish')
    file.close()
    return collections.Counter(lst)


# Uncomment to create a new pickling file that will hold the dictionary of words and the sentences in which they occur
#if __name__ == '__main__':

    #tok_utterance_list = [ nltk.word_tokenize(sent.lower(), language='spanish') for sent in
    #                   make_list_utterances(os.path.join(os.getcwd(),'corpus_files\\AAMegaDoc.txt'))]

    #vocab = make_vocabulary_list(os.path.join(os.getcwd(),'corpus_files\\AAMegaDoc.txt'))

    #make_freq_dict(vocab, tok_utterance_list)
