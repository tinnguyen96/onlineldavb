# topicmodelvb.py

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi
import warnings

import corpus
from corpus import parse_doc_list
from corpus import make_vocab
from corpus import split_document

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    Inputs:
        alpha: K x V, the Dirichlet parameters are stored in rows.
    Outputs:
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class _TopicModel:
    """
    Skeleton for SVI training of topic models (LDA 1/K or SB-LDA).
    """

    def __init__(self, vocab, K, topicpath, D, alpha0, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. 
        topicpath: Path to some pre-trained topics' Dirichlet parameters. User
        alpha0: for LDA 1/K, alpha = alpha0/K is the hyperparameter for prior on topic 
            proportions theta_d. For SB-LDA, alpha0 is governs the stick-breaking
            Beta(1,alpha0). 
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Remarks:
            User should make sure the topics from topicpath are compatible with 
            K and vocab.
        """
        t0 = time.time()
        self._vocab, self._idxtoword = make_vocab(vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda) if topicpath is not None
        if (topicpath is None):
            self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        else:
            self._lambda = n.loadtxt(topicpath)
            assert self._lambda.shape==(self._K,self._W), "Wrong shape of topics"
            print("Successfully loaded topics from %s" %topicpath)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        t1 = time.time()
        print(("Time to initialize topic model is %.2f" %(t1-t0)))
        
        return

    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
    
        Returns variational parameters for per-document topic proportions.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        varparams, sstats = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(wordids))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return varparams

    def log_likelihood_one(self, wordobs_ids, wordobs_cts, wordho_ids, \
                      wordho_cts):
        """
        Inputs:
            wordobs_ids: list, index in vocab of unique observed words
            wordobs_cts: list, number of occurences of each unique observed word
            wordho_ids: list, index in vocab of held-out words
            wordho_cts: list, number of occurences of each unique held-out word
        Outputs:
            average log-likelihood of held-out words for the given document
        """
        # theta_means should be 1 x self._K
        theta_means = self.theta_means(wordobs_ids, wordobs_cts)
        # lambda_sums should be self._K x 1
        lambda_sums = n.sum(self._lambda, axis=1) 
        # lambda_means should be self._K x self._W, rows suming to 1
        lambda_means = self._lambda/lambda_sums[:, n.newaxis] 
        Mho = list(range(0,len(wordho_ids)))
        proba = [wordho_cts[i]*n.log(n.dot(theta_means,lambda_means[:,wordho_ids[i]])) \
                for i in Mho]
        # average across all held-out words
        tot = sum(wordho_cts)
        return sum(proba)/tot

    def log_likelihood_docs(self, wordids, wordcts):
        """
        Inputs:
            wordids: list of lists
            wordcts: list of lists
        Outputs:
        """ 
        t0 = time.time()
        M = len(wordids)
        log_likelihoods = []
        for i in range(M):
            docids = wordids[i] # list 
            doccts = wordcts[i] # list
            # only evaluate log-likelihood if non-trivial document
            if len(docids) > 1:
                wordobs_ids, wordobs_cts, wordho_ids, wordho_cts = \
                    split_document(docids, doccts)
                doc_likelihood = \
                    self.log_likelihood_one(wordobs_ids, wordobs_cts, wordho_ids, wordho_cts)
                log_likelihoods.append(doc_likelihood)
        t1 = time.time()
        # print("Time taken to evaluate log-likelihood %.2f" %(t1-t0))
        return n.mean(log_likelihoods)
    
class LDA(_TopicModel):
    """
    Inherit _TopicModel to train LDA 1/K. 
    """

    def __init__(self, vocab, K, topicpath, D, alpha0, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        topicpath:
        D: Total number of documents in the population. 
        alpha0:
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._alpha = alpha0/K
        _TopicModel.__init__(self, vocab, K, topicpath, D, alpha0, eta, tau0, kappa)
        
        return
    
    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        # each gamma[:,:] has mean 1 and variance 0.01
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        converged = False
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 200):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            # might have exited coordinate ascent without convergence
            """
            if (not converged):
                print("Coordinate ascent in E-step didn't converge")
                print("Last change in gammad %f" %meanchange)
            """
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return (gamma,sstats) 

    def theta_means(self, wordobs_ids, wordobs_cts):
        """
        Inputs:
            wordobs_ids = list
            wordobs_cts = list
        Outputs:
            Report E(q(theta(k)) across topics, where q(theta) is variational 
            approximation of the new document's topic proportions.
        """
        # do E-step for the document represented by the observed words
        # gamma should be 1 x self._K
        gamma, _ = self.do_e_step([wordobs_ids],[wordobs_cts]) 
        # q(theta|gamma) is Dirichlet, so marginal means are average of Dirichlet parameters
        theta = gamma/n.sum(gamma) 
        theta = theta.flatten(order='C') 
        return theta

class SB_LDA(_TopicModel):
    """
    Inherit _TopicModel to train SB-LDA at level K. 
    """

    def __init__(self, vocab, K, topicpath, D, alpha0, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        topicpath: 
        D: Total number of documents in the population. 
        alpha0: Hyperparameter of the stick-breaking weights Beta(1,alpha0).
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        """
        self._alpha0 = alpha0
        _TopicModel.__init__(self, vocab, K, topicpath, D, alpha0, eta, tau0, kappa)
         # for updating tau given phi
        mask = n.zeros((self._K, self._K))
        for i in range(self._K):
            for j in range(self._K):
                mask[i,j] = int(j > i)
        self._fmask = mask # size (self._K, self._K)
        # for updating phi given tau
        self._bmask = mask.transpose()
        """
        print("Mask for updating tau given phi")
        print(self._fmask)
        print("Mask for updating phi given tau")
        print(self._bmask)
        """
        return
    
    def init_phi(self, ids):
        """
        Inputs:
            ids:
        Outputs:
            initialize phi as if all Elogthetad is equal to each other i.e. 
            only considering effect of topics rather than topic proportions.
        """
        Elogbetad = self._Elogbeta[:, ids] # (self._K, len(ids))
        logphi = Elogbetad # size (self._K, len(ids))
         # normalize across rows
        unormphi = n.exp(logphi)
        phinorm = n.sum(unormphi, axis=0)+1e-100 # size should be 1 x len(cts)
        phi = unormphi/phinorm[n.newaxis, :]
        return phi

    def init_tau(self, batch_size):
        """
        Inputs:
            batch_size = number of documents being processed 
        Outputs:
            initialize tau to be uninformative about theta
        Remarks:
            try out some options in initialize tau
        """
        tau1 = n.random.gamma(100., 1./100., (batch_size, self._K))
        tau1[:,self._K-1] = 1 # corner case
        tau2 = n.random.gamma(100*self._alpha0, 1./100., (batch_size, self._K))
        tau2[:,self._K-1] = 0 # corner case
        return tau1, tau2
    
    def opt_phi(self, tau1d, tau2d, ids):
        """
        Inputs:
            tau1d:
            tau2d:
            ids:
        Outputs:
        """
        Elogbetad = self._Elogbeta[:, ids] # (self._K, len(ids))
        Elogpm1pd = dirichlet_expectation(n.column_stack((tau1d,tau2d)))
        Elogpd = Elogpm1pd[:,0] # shape (self._K,). 
        Elogm1pd = Elogpm1pd[:,1] # shape (self._K,). Last value is -Inf since Beta(1,0), need to fix.
        Elogm1pd[self._K-1] = 0
        """
        print("Elogm1pd")
        print(Elogm1pd)
        """
        # for now, explicitly represent optimal phi to ensure correctness 
        Elogthetad = Elogpd + n.dot(self._bmask, Elogm1pd) # shape (self._K,)
        """
        print("Elogthetad")
        print(Elogthetad)
        """
        logphi = Elogthetad[:, n.newaxis] + Elogbetad # size (self._K, len(cts))
        # normalize across rows
        unormphi = n.exp(logphi)
        phinorm = n.sum(unormphi, axis=0)+1e-100 # size should be 1 x len(cts)
        phi = unormphi/phinorm[n.newaxis, :]
        return phi
    
    def opt_tau(self, phi, cts):
        """
        Inputs:
            phi:
            cts:
        Outputs:
        """
        tau1d = 1 + n.dot(phi, cts) # careful, dot of 2-D array with list!
        tau2d = self._alpha0 + n.dot(n.dot(self._fmask, phi), cts) # careful, dot of 2-D array with list
        tau1d[self._K - 1] = 1
        tau2d[self._K - 1] = 0
        return (tau1d, tau2d)
    
    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        # Initialize the variational distribution q(pi|tau) for
        # the mini-batch
        tau1, tau2 = self.init_tau(batchD) # each has size batchD x self._K
        sstats = n.zeros(self._lambda.shape) # shape (self._K, self._W)
        
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d] # list
            cts = wordcts[d] # list
            tau1d, tau2d = tau1[d, :], tau2[d, :] # each has size (self._K,)
            phi = self.opt_phi(tau1d, tau2d, ids)
            # Iterate between tau and phi until convergence
            converged = False
            for it in range(0, 200):
                lasttau1 = tau1d
                lasttau2 = tau2d
                tau1d, tau2d = self.opt_tau(phi, cts)
                phi = self.opt_phi(tau1d, tau2d, ids)
                # If tau hasn't changed much, we're done.
                meanchange = 0.5*n.mean(abs(lasttau1 - tau1d)) + 0.5*n.mean(abs(lasttau2 - tau2d))
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            
            # might have exited coordinate ascent without convergence
            """
            if (not converged):
                print("Coordinate ascent in E-step didn't converge")
                print("Last change in taud %f" %meanchange)
            """
            tau1[d, :] = tau1d
            tau2[d, :] = tau2d
            sstats[:, ids] += n.multiply(phi,cts)

        return ((tau1, tau2), sstats)
    
    def debug_e_step(self, wordids, wordcts):
        """
        Same functionality as do_e_step, but with more prints to aid
        debugging: print initial guess of topic proportions, and that
        at convergence.
        """
        batchD = len(wordids)

        # Initialize the variational distribution q(pi|tau) for
        # the mini-batch
        tau1, tau2 = self.init_tau(batchD) # each has size batchD x self._K
        sstats = n.zeros(self._lambda.shape) # shape (self._K, self._W)
        
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d] # list
            cts = wordcts[d] # list
            tau1d, tau2d = tau1[d, :], tau2[d, :] # each has size (self._K,)
            phi = self.opt_phi(tau1d, tau2d, ids)
            # Iterate between tau and phi until convergence
            converged = False
            for it in range(0, 200):
                lasttau1 = tau1d
                lasttau2 = tau2d
                tau1d, tau2d = self.opt_tau(phi, cts)
                phi = self.opt_phi(tau1d, tau2d, ids)
                # If tau hasn't changed much, we're done.
                meanchange = 0.5*n.mean(abs(lasttau1 - tau1d)) + 0.5*n.mean(abs(lasttau2 - tau2d))
                if (meanchange < meanchangethresh):
                    converged = True
                    break
            tau1[d, :] = tau1d
            tau2[d, :] = tau2d
            sstats[:, ids] += n.multiply(phi,cts)

        return ((tau1, tau2), sstats)

    def theta_means(self, wordobs_ids, wordobs_cts):
        """
        Inputs:
            wordobs_ids = list
            wordobs_cts = list
        Outputs:
            Report E(q(theta(k)) across topics, where q(theta) is variational 
            approximation of the new document's topic proportions.
        Remarks:
        """
        taus, _ = self.do_e_step([wordobs_ids],[wordobs_cts]) 
        tau1, tau2 = taus[0], taus[1] # each's shape is 1 x self._K
        # theta(k) = p(k) x prod_{i=1}^{k-1} (1-p(i)), each p(i) Beta(tau1(i), tau2(i))
        # and they are independent because of mean-field.
        Ep = tau1/(tau1+tau2)
        Em1p = 1-Ep # last value is 0 since theta(K) is Beta(1,0)
        """
        print(tau1)
        print(tau2)
        print(Ep)
        """
        Em1p[0,self._K-1] = 1 # hack
        cumu = n.cumprod(Em1p, axis=1) # shape (self._K,)
        """
        print("Em1p shape")
        print(Em1p.shape)
        """
        ratiop = Ep/Em1p # shape (1 x self._K)
        """
        print("ratiop shape")
        print(ratiop.shape)
        """
        """
        print(cumu[0,:(self._K-1)].shape)
        """
        theta = n.multiply(ratiop, cumu) # shape (1 x self._K)
        """
        print(theta)
        """
        return theta
    
def main():
    # examine effect of SB-LDA E-step's on a document
    ## load topics
    K = 100
    inroot = "wiki10k"
    infile = inroot + "_wordids.csv"
    with open(infile) as f:
        D = sum(1 for line in f)
    vocab = open('./dictnostops.txt').readlines()
    topicpath = "results/sbldaK100_D16_wiki10k_wiki1k/lambda-100.dat"
    tm = SB_LDA(vocab, K, topicpath, D, 1, 0.01, 1024., 0.7)
    
    ## E-step on a document, plotting initial guess of topic proportions 
    ## as well as their convergence
    (wordids, wordcts) = \
            corpus.get_batch_from_disk(inroot, D, 1)
    tm.debug_e_step(wordids, wordcts)
    
    return

if __name__ == '__main__':
    main()
