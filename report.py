import numpy as np
import matplotlib 

def plotldaLL(K, inroot, heldoutroot, expset):
    """
    Inputs:
        expset: list of experimental seeds
    Outputs:
    
    Remarks: 
        all experiments are expected to run until completion i.e. 
        the same number of iterations.
    """
    
    # Load and compute average LDA 1/K
    ldadir = "ldaK" + str(K) + "_" + inroot + "_" + heldoutroot
    ldaLL = []
    ## load experiments
    for seed in expset:
        ldapath = ldadir + "/_" + str(seed) + ".csv"
        result = np.loadtxt(ldapath)
        maxbatchcount = len(result[:,0])
        ldaLL.append(result[:,2])
    ldaLL = np.array(ldaLL)
    ldaavg = np.mean(ldaLL, axis=1) # 
    ldaerr = np.std(ldaLL, axis=1) # 
    
    fig = plt.figure()
    plt.errorbar(range(maxbatchcount), ldaavg, yerr=ldaerr, fmt='-o')
    plt.title('Held-out log-likelihood vs number of mini-batches trained')
    plt.xlabel('Number of mini-batches')
    plt.ylabel('Held-out log-likelihood')
    plt.show()
    
    return 

def plotsbLL(K, inroot, heldoutroot, expset):
    return