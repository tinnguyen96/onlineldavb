import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb
import wikirandom
import corpus

def main():
    """
    Load a wikipedia corpus in batches from disk and run LDA.
    """

    # The rootname, for instance wiki10k
    inroot = sys.argv[1]
    infile = inroot + "_wordids.csv"
    
    # The total number of documents in Wikipedia
    with open(infile) as f:
        D = sum(1 for line in f)
    print("Corpus has %d documents" %D)

    # Set random seed for replicability
    seed = int(sys.argv[2])
    numpy.random.seed(seed)

    # The number of documents to analyze each iteration
    batchsize = 16

    # The number of topics
    K = 100

    max_iter = 100

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 0.01, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, documentstoanalyze):
        # Load a random batch of articles from disk
        (wordids, wordcts) = \
            wikirandom.get_batch_from_disk(inroot, D, batchsize)
        # Give them to LDA
        (gamma, bound) = olda.update_lambda(wordids, wordcts)
        # Compute an estimate of held-out perplexity
        perwordbound = bound * len(wordids) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

if __name__ == '__main__':
    main()
