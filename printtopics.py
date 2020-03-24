"""
# printtopics.py: Prints the words that are most prominent in a set of topics.
Command-line arguments:
- dictionary, for instance dictnostops.txt
- topics i.e. Dirichlet parameters, for instance 
- whether to report marginal means

Example call:
python printtopics.py dictnostops.txt ldaK100_D16_wiki10k_wiki1k/lambda-100.dat 1
"""

import sys, os, re, random, math, urllib.request, urllib.error, urllib.parse, time, pickle
import numpy

def main():
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    vocab = open(sys.argv[1]).readlines()
    testlambda = numpy.loadtxt(sys.argv[2])
    normalized = bool(int(sys.argv[3]))
    threshold = 10

    sigtopics = 0
    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        # report marginal probabilities instead of Dirichlet parameters
        if (normalized):
            lambdak = lambdak / sum(lambdak)
        temp = list(zip(lambdak, list(range(0, len(lambdak)))))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        # plot topics with parameters exceeding the threshold
        if (temp[0][0] >= threshold):
            sigtopics += 1
            print('topic %d:' % (k))
            # feel free to change the "53" here to whatever fits your screen nicely.
            for i in range(0, 20):
                print('%20s  \t---\t  %.4f' % (vocab[temp[i][1]], temp[i][0]))
            print()
    print('Printed %d significant topics at threshold %.2f' %(sigtopics, threshold))

if __name__ == '__main__':
    main()
