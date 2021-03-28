import numpy as np
import re
import string
import collections
import random
from scipy.special import gammaln, psi, polygamma
from functools import reduce
from warnings import warn

def parse_doc(doc,vocab):
    """
    Parse a single document. 
    Arguments:
    doc: document string
    vocab: a dictionary that maps words to integers
    Output:
    A dictionary, where the keys are words appeared in the doc, labeled with the integers in the vocab dictionary (the set of $\tilde{w_n}$), 
        and the values are counts of the words.
    The words that are not in vocab will be ignored.
    """
    doc=doc.lower()
    doc=re.sub(r'-',' ',doc)
    doc=re.sub(r' +',' ',doc) # turn multiple spaces into a single space
    doc=re.sub(r'[^a-z ]','',doc) # remove anything that is not a-z or space
    words=doc.split()
    word_vocab=[vocab.get(word,-1) for word in words]
    words_dict=collections.Counter(word_vocab)
    del words_dict[-1] # ignore the words outside the vocabulary
    #wordid=words_dict.keys()
    #wordcnt=words_dict.values()
    return words_dict