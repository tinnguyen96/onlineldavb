March 19, 2020:
    The articles in wiki10k and wiki1k are not guaranteed to be disjoint from each other.
    
March 20, 2020:
    the function that is hardest to convert to Python 3 from Python 2 is wikirandom.py,
    so we leave as is. We also leave onlinewikipedia.py as Python 2 since we don't use it.
    
March 21, 2020:
    Currently representing the variational parameter of per-word topic assignment explicitly
    in SB-LDA's do-e-step. Correctness is the priority now. Later, to save time and memory, 
    might switch to implicit representation.  