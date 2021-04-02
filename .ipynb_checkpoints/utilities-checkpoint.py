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
    A list of tuples, where for each tuple, the first element is a word appeared in the doc, labeled with the integers in the vocab dictionary (the set of $\tilde{w_n}$), 
        and the second element is count of the words.
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
    return sorted(words_dict.items())

def make_vocab_from_docs(docs):
    """
    Make a dictionary that contains all words from the docs. The order of words is arbitrary.
    docs: iterable of documents
    """
    vocab_words=set()
    for doc in docs:
        doc=doc.lower()
        doc=re.sub(r'-',' ',doc)
        doc=re.sub(r' +',' ',doc) # turn multiple spaces into a single space
        doc=re.sub(r'[^a-z ]','',doc) # remove anything that is not a-z or space
        words=set(doc.split())
        vocab_words=vocab_words.union(words)
        vocab=dict(zip(vocab_words,range(len(vocab_words))))
    return vocab

def make_data(docs):
    """
    Make the input for variational_em function from docs.
    """
    vocab=make_vocab_from_docs(docs)
    word_dicts=list(map(lambda x: parse_doc(x,vocab),docs))
    Nd=list(map(len,docs))
    M,V=len(docs),len(vocab)
    return (vocab,word_dicts,Nd,M,V)

def loglik(alpha,gamma_list,M,k):
    """
    Calculate $L_{[\alpha]}$ defined in A.4.2
    """
    psi_sum_gamma=np.array(list(map(lambda x: psi(np.sum(x)),gamma_list))).reshape((M,1)) # M*1 
    psi_gamma=psi(np.array(gamma_list)) # M*k matrix
    L=M*gammaln(np.sum(alpha)-np.sum(gammaln(alpha)))+np.sum((psi_gamma-psi_sum_gamma)*(alpha.reshape((1,k))-1))
    return L
    