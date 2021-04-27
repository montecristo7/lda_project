import pandas as pd
import numpy as np
import re
from scipy.special import psi  # gamma function utils
from pprint import pprint
import gensim.corpora as corpora
from gensim.corpora import Dictionary


stop_words = ["us", "also", "may", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

## Utils and Helper Class

def tf(docs):
    """
    This function is used to calculate the document-term matrix and id2word mapping
    """
    # Clean up the text
    docsc_clean = {}
    total_term = []
    for key, val in enumerate(docs):
        val_clean = re.findall(r'[a-z]+', val.lower())
        val_clean = [i for i in val_clean if i not in stop_words]
        docsc_clean[f'd{key}'] = val_clean
        total_term += val_clean

    total_term_unique = sorted(set(total_term))
    id2word = {idx: word for  idx, word in enumerate(total_term_unique)}

    # Count the number of occurrences of term i in document j
    for key, val in docsc_clean.items():
        word_dir = dict.fromkeys(total_term_unique, 0)
        for word in val:
            word_dir[word] += 1
        docsc_clean[key] = word_dir

    tf_df = pd.DataFrame.from_dict(docsc_clean, orient='index')

    return tf_df, id2word

def dirichlet_expectation(sstats):
    if len(sstats.shape) == 1:
        return psi(sstats) - psi(np.sum(sstats))
    else:
        return psi(sstats) - psi(np.sum(sstats, 1))[:, np.newaxis]
    
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class LdaState:
    def __init__(self, eta, shape, dtype=np.float32):
        """
        Parameters
        ----------
        eta : numpy.ndarray
            The prior probabilities assigned to each term.
        shape : tuple of (int, int)
            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).
        dtype : type
            Overrides the numpy array default types.

        """
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def get_lambda(self):
        """Get the parameters of the posterior over the topics, also referred to as "the topics".

        Returns
        -------
        numpy.ndarray
            Parameters of the posterior probability over topics.

        """
        return self.eta + self.sstats

    def get_Elogbeta(self):
        """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        numpy.ndarray
            Posterior probabilities for each topic.
        """
        return dirichlet_expectation(self.get_lambda())

    def blend(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted average for the sufficient statistics.

        The number of documents is stretched in both state objects, so that they are of comparable magnitude.
        This procedure corresponds to the stochastic gradient update from
        `Hoffman et al. :"Online Learning for Latent Dirichlet Allocation"
        <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_, see equations (5) and (9).

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats
        self.numdocs = targetsize
        
        
def my_lda_func(corpus, num_topics, id2word, random_state=10,  passes=1, num_words=10,
                iterations=50, gamma_threshold=0.001, dtype=np.float32,  chunksize=100, topics_only=True, verbose=False):
    num_terms = len(id2word)

    alpha = np.array( [1.0 / num_topics for i in range(num_topics)], dtype=dtype)

    eta = np.array( [1.0 / num_topics for i in range(num_terms)], dtype=dtype)

    rand  = np.random.RandomState(random_state)

    model_states = LdaState(eta, (num_topics, num_terms), dtype=dtype)
    model_states.sstats = rand.gamma(100., 1. / 100., (num_topics, num_terms))

    expElogbeta = np.exp(dirichlet_expectation(model_states.sstats))


    # Update
    lencorpus = len(corpus)
    chunksize = min(lencorpus, chunksize)
    model_states.numdocs += lencorpus
    num_updates = 0

    for pass_ in range(passes):
        all_chunks = chunks(corpus, chunksize)
        gamma_by_chunks = []
        for chunk_no, chunk in enumerate(all_chunks):
            other = LdaState(eta, (num_topics, num_terms), dtype=dtype)
            # Do estep
            if len(chunk) > 1:
                if verbose:
                    print(f'performing inference on a chunk of {len(chunk) } documents')
            else:
                raise

            # Initialize the variational distribution q(theta|gamma) for the chunk
            gamma = rand.gamma(100., 1. / 100., (len(chunk), num_topics)).astype(dtype, copy=False)
            tmpElogtheta = dirichlet_expectation(gamma)
            tmpexpElogtheta = np.exp(tmpElogtheta)
            sstats = np.zeros_like(expElogbeta, dtype=dtype)
            converged = 0

            # Now, for each document d update that document's gamma and phi
            epsilon = 1e-7

            for d, doc in enumerate(chunk):
                ids = [idx for idx, _ in doc]
                cts = np.fromiter((cnt for _, cnt in doc), dtype=dtype, count=len(doc))
                gammad = gamma[d, :]
                Elogthetad = tmpElogtheta[d, :]
                expElogthetad = tmpexpElogtheta[d, :]
                expElogbetad = expElogbeta[:, ids]

                # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
                # phinorm is the normalizer.
                phinorm = np.dot(expElogthetad, expElogbetad) + epsilon

                for _ in range(iterations):
                    lastgamma = gammad
                    # We represent phi implicitly to save memory and time.
                    # Substituting the value of the optimal phi back into
                    # the update for gamma gives this update. Cf. Lee&Seung 2001.
                    gammad = alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                    Elogthetad = dirichlet_expectation(gammad)
                    expElogthetad = np.exp(Elogthetad)
                    phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
                    # If gamma hasn't changed much, we're done.
                    meanchange = np.mean(np.abs(gammad - lastgamma))
                    if meanchange < gamma_threshold:
                        converged += 1
                        break
                gamma[d, :] = gammad
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)
                gamma_by_chunks.append(gamma)

            if len(chunk) > 1:
                if verbose:
                    print(f"{converged}/{len(chunk)} documents converged within {iterations} iterations")

            sstats *= expElogbeta

            other.sstats += sstats
            other.numdocs += gamma.shape[0]

            # Do mstep
            if verbose:
                print('Update topics')
            previous_Elogbeta = model_states.get_Elogbeta()
            rho = pow(1 + pass_ + (num_updates / chunksize), -0.5)
            model_states.blend(rho, other)

            current_Elogbeta = model_states.get_Elogbeta()
            #Propagate the states topic probabilities to the inner object's attribute.
            expElogbeta = np.exp(current_Elogbeta)

            diff = np.mean(np.abs(previous_Elogbeta.ravel() - current_Elogbeta.ravel()))
            if verbose:
                print(f"topic diff {diff}")
            num_updates += other.numdocs

    shown = []
    topic = model_states.get_lambda()

    for i in range(num_topics):
        topic_ = topic[i]
        topic_ = topic_ / topic_.sum()  # normalize to probability distribution
        bestn = topic_.argsort()[-num_words:][::-1]

        topic_ = [(id2word[id], topic_[id]) for id in bestn]
        topic_ = ' + '.join('%.3f*"%s"' % (v, k) for k, v in topic_)
        shown.append((i, topic_))

    if topics_only:
        return shown
    else:
        return shown,gamma_by_chunks
    

    
