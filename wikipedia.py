import pickle, string, numpy, getopt, sys, random, time, re, pprint

import topicmodel
import corpus

def main():
    """
    Load a wikipedia corpus in batches from disk and run LDA.
    """

    # The rootname, for instance wiki10k
    inroot = sys.argv[1]
    infile = inroot + "_wordids.csv"

    # For instance, wiki1k
    heldoutroot = sys.argv[2]
    heldoutfile = heldoutroot + "_wordids.csv"
    
    with open(infile) as f:
        D = sum(1 for line in f)
    print(("Training corpus has %d documents" %D))

    with open(heldoutfile) as f:
        D_ = sum(1 for line in f)
    print(("Held-out corpus has %d documents" %D_))

    # Set random seed for replicability
    seed = int(sys.argv[3])
    numpy.random.seed(seed)

    # The number of documents to analyze each iteration
    batchsize = 16

    # The number of topics
    K = 100

    max_iter = 100

    # Our vocabulary
    vocab = open('./dictnostops.txt').readlines()
    W = len(vocab)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = topicmodelvb.OnlineLDA(vocab, K, D, 1./K, 0.01, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, max_iter):
        # Load a random batch of articles from disk
        (wordids, wordcts) = \
            corpus.get_batch_from_disk(inroot, D, batchsize)
        # Give them to LDA
        (gamma, bound) = olda.update_lambda(wordids, wordcts)
        # Compute average log-likelihood on held-out corpus
        (howordids,howordcts) = \
            corpus.get_batch_from_disk(heldoutroot, D_, None)
        LL = olda.log_likelihood_docs(howordids,howordcts)
        print('%d:  rho_t = %f,  held-out log-likelihood = %f' % \
            (iteration, olda._rhot, LL))

if __name__ == '__main__':
    main()
