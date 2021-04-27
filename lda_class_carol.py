import numpy as np
import re
import string
import collections
import random
from scipy.special import gammaln, psi, polygamma
from functools import reduce
from warnings import warn
import utilities

class BaseLDA(object):
    def __init__(self, docs):
        self.docs=docs
    def lda(self,num_topics):
        raise ValueError('Method not implemented.')
        
class LDA2(BaseLDA):
    def __init__(self, docs):
        self.M=len(docs)
        self.vocab=None
        self.V=-1
        self.topics=None
        self.gamma=None
        super().__init__(docs)

    def make_vocab_from_docs(self):
        """
        Make a dictionary that contains all words from the docs. The order of words is arbitrary.
        docs: iterable of documents
        """
        vocab_words=set()
        for doc in self.docs:
            doc=doc.lower()
            doc=re.sub(r'-',' ',doc)
            doc=re.sub(r' +',' ',doc) # turn multiple spaces into a single space
            doc=re.sub(r'[^a-z ]','',doc) # remove anything that is not a-z or space
            words=set(doc.split())
            vocab_words=vocab_words.union(words)
            vocab=dict(zip(vocab_words,range(len(vocab_words))))
        self.vocab=vocab
        self.V=len(vocab)
        return vocab
    def parse_doc(self,doc,vocab):
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
    def e_step(self,N,k,V,alpha,beta,word_dict,conv_threshold,max_iter,verbose=False):
        """
        Variational inference algorithm for document-specific parameters of a single doc in LDA with the equivalent class representation.
        Arguments:
        N: number of words
        k: number of topics
        V: length of vocabulary
        alpha: corpus-level Dirichlet parameter, k-vector
        beta: corpus-level multinomial parameter, k * V matrix
        word_dict: word_dict from parse_doc
        conv_threshold: threshold for convergence
        max_iter: maximum number of iterations
        Output:
        A tuple of document specific optimizing parameters $(\gamma^*, \phi^*)$ obtained from variational inference.  
        First element: $\gamma^*$, k-vector
        Second element: the second sum in Eq(9), k*V matrix
        """
        conv=False
        wordid=list(map(lambda x:x[0],word_dict))
        wordcnt=list(map(lambda x:x[1],word_dict))
        phi0=np.full(shape=(len(wordid),k),fill_value=1/k) # phi_tilde 
        phi1=np.zeros(shape=(len(wordid),k))
        gamma0=alpha+N/k
        for it in range(max_iter):
            print(it)
            for j in range(len(wordid)):
                # the jth row of phi1 corresponds to the word labelled as wordid[j]
                for i in range(k):
                    #phi1[j,i]=beta[i,wordid[j]]*np.exp(psi(gamma0[i]))*wordcnt[j]
                    phi1[j,i]=beta[i,wordid[j]]*np.exp(psi(gamma0[i]))
                phi1[j,]=phi1[j,]/np.sum(phi1[j,])
            gamma1=alpha+np.sum(phi1*(np.array(wordcnt).reshape((-1,1))),axis=0)
            #gamma1=alpha+np.sum(phi1,axis=0)
            # stop if gamma has converged
            if np.max(np.abs((gamma0-gamma1)))<conv_threshold:
                conv=True
                break
            gamma0=gamma1
            phi0=phi1 
        if not conv and verbose:
            warn('Variational inference has not converged. Try more iterations.')
        suff_stat=np.zeros(shape=(V,k))
        suff_stat[wordid,]=phi1*(np.array(wordcnt).reshape((-1,1)))
        return (gamma1,suff_stat.T) 
    def m_step_exp(self,M,k,V,suff_stat_list,gamma_list,alpha0,conv_threshold,max_iter,verbose=False):
        """
        M-step in variational EM, maximizing the lower bound on log-likelihood w.r.t. alpha and beta. (Section 5.3)
        Arguments:
        M: number of documents in the corpus
        k: number of topics
        V: length of vocab
        suff_stat_list: M-list of sufficient statistics (k * V matrices), one for each doc
        gamma_list: M-list of gamma's (k-vectors), one for each doc
        alpha0: initialization of alpha in Newton-Raphson
        conv_threshold: convergence threshold in Newton-Raphson
        max_iter: maximum number of iterations in Newton-Raphson
        Output:
        A 2-tuple. 
        First element: beta (k*V matrix)
        Second element: alpha (k*1)
        """
        alphalist=[alpha0]
        ll=[]
        ll0=conv_threshold
        conv=False
        # update beta
        beta=reduce(lambda x,y: x+y, suff_stat_list)
        beta=beta/np.sum(beta,axis=1).reshape((-1,1))
        # update alpha (Newton-Raphson)
        alpha0=alpha0.reshape((k,1))
        psi_sum_gamma=np.array(list(map(lambda x: psi(np.sum(x)),gamma_list))).reshape((M,1)) # M*1 
        psi_gamma=psi(np.array(gamma_list)) # M*k matrix
        for it in range(max_iter):
            print(it)
            a0=np.log(alpha0)
            psi_sum_alpha=psi(np.sum(alpha0))
            poly_sum_alpha=polygamma(1,np.sum(alpha0))
            g=M*(psi_sum_alpha-psi(alpha0)).reshape((k,1))+np.sum(psi_gamma-psi_sum_gamma,axis=0).reshape((k,1))*alpha0.reshape((k,1)) # k*1
            H=alpha0@alpha0.T*M*poly_sum_alpha+np.diag(g.reshape((k,))+1e-10-(alpha0**2*M*polygamma(1,alpha0)).reshape((k,)))
            a1=a0-np.linalg.inv(H)@g
            alpha1=np.exp(a1)
            ll1=utilities.loglik(alpha1,gamma_list,M,k)
            ll.append(ll1)
            if np.abs((ll1-ll0)/(1+abs(ll0)))<conv_threshold:
                #print('newton finished at iteration',it)
                conv=True
                break
            alpha0=alpha1
            a0=np.log(alpha0)
            alphalist.append(alpha1)
            ll0=ll1
        if not conv and verbose:
            warn('Newton-Raphson has not converged. Try more iterations.')
        return (beta,alpha1,ll,alphalist)
    def variational_em_all(self,Nd,alpha0,beta0,word_dicts,vocab,M,k, conv_threshold,max_iter,npass,m_func=m_step_exp,verbose=False):
        """
        Input:
        Nd: list of length of documents 
        alpha0: initialization of alpha
        beta0: initialization of beta. DO NOT initialize with identical rows!
        word_dicts: list of word_dict of documents, in the same order as N
        vocab: vocabulary
        M: number of documents
        k: number of topics
        """
        V=len(vocab)
        for it in range(npass):
            #
            e_estimates=list(map(lambda x,y: self.e_step(x,k,V,alpha0,beta0,y,conv_threshold=conv_threshold,max_iter=max_iter), Nd,word_dicts,verbose=verbose))
            gamma_list=list(map(lambda x:x[0],e_estimates))
            suff_stat_list=list(map(lambda x:x[1],e_estimates))
            m_estimates=m_func(self,M,k,V,suff_stat_list,gamma_list,alpha0,conv_threshold=conv_threshold,max_iter=max_iter,verbose=verbose)
            alpha1=m_estimates[1]
            beta1=m_estimates[0]
            if np.max(np.abs(beta1-beta0))<conv_threshold:
                #print('vem finished at iteration',it)
                break
            alpha0=alpha1.reshape(k)
            beta0=beta1
        return (alpha0,beta0,gamma_list,suff_stat_list)
    def lda(self,num_topics,num_words=None,alpha0='rand_init',beta0='rand_init',conv_threshold=1e-3,max_iter=int(1e3),npass=int(1e3),verbose=False):
        """Fit LDA to the corpus with given number of topics. Returns the words with highest probablity in each topic."""
        vocab=self.make_vocab_from_docs()
        word_dicts=list(map(lambda x: self.parse_doc(x,vocab),self.docs))
        Nd=list(map(len,self.docs))
        k,M,V=num_topics,len(self.docs),len(self.vocab)
        if alpha0=='rand_init':
            np.random.seed(1)
            alpha0=np.exp(np.random.random(k))
        if beta0=='rand_init':
            np.random.seed(3)
            str_whole=reduce(lambda x,y:x+' '+y, self.docs)
            pd=self.parse_doc(str_whole,vocab)
            #beta0=np.array([w[1] for w in pd]*k).reshape((k,V))
            beta0=np.random.random((k,V))
            beta0=beta0/np.sum(beta0,axis=1).reshape((-1,1))
        vem=self.variational_em_all(Nd,alpha0,beta0,word_dicts,vocab,M,k, conv_threshold,max_iter,npass,verbose=verbose)
        beta_post=vem[1]
        topics=[dict(zip(list(vocab.keys()),beta_post[i,:])) for i in range(k)]
        topics=[sorted(topic.items(),key=lambda x:x[1],reverse=True) for topic in topics]
        self.topics=topics
        self.gamma=vem[2]
        if num_words:
            return [topic[0:num_words] for topic in topics]
        else: 
            return topics
    

