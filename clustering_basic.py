import numpy as np
from clustering_grassmannian import plusplus_initialization

def kmeans_clustering(X,K,seed=None, data_gr=True):
    if data_gr: X = X[:,:,0] # from data_gr to data_sphere
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=K, n_init=10).fit(X)
    return (model.cluster_centers_, model.labels_, model.score(X))


def diametrical_clustering(X,K,seed=None,max_iter=10000,num_repl=1,init=None,call=0,tol=1e-16,data_gr=True):
    if seed is not None: np.random.seed(seed)
    if data_gr: X = X[:,:,0] # from data_gr to data_sphere
    """
    Diametrical clustering algorithm for clustering data on the sign-symmetric unit (hyper)sphere.
    Originally proposed in "Diametrical clustering for identifying
    anti-correlated gene clusters" by Dhillon IS et al., 2003 (Bioinformatics).
    Current version implemented from "The multivariate Watson distribution: 
    Maximum-likelihood estimation and other aspects" by Sra S & Karp D, 2012, Journal of Multivariate Analysis

    Input:
        X: data matrix (n,p)
        K: number of clusters
        max_iter: maximum number of iterations
        num_repl: number of repetitions
        init: initialization method. Options are '++' (or 'plusplus' or 'diametrical_clustering_plusplus'), 'uniform' (or 'unif')
        call: number of times the function has been called recursively
        tol: tolerance for convergence
    Output:
        C: cluster centers
        part: partition
        obj: objective function value
    """

    n,p = X.shape

    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector

    # loop over the number of repetitions
    for _ in range(num_repl):
        if init is None or init=='++' or init=='plusplus' or init == 'diametrical_clustering_plusplus':
            C,_,_ = plusplus_initialization(X,K,dist='diametrical')
        elif init=='uniform' or init=='unif':
            C = np.random.uniform(size=(p,K))
            C = C/np.linalg.norm(C,axis=0)
        
        iter = 0
        obj = [] # objective function
        partsum = np.zeros((max_iter,K))
        while True:
            # "E-step" - compute the similarity between each point and each cluster center
            sim = (X@C)**2
            maxsim = np.max(sim,axis=1) # find the maximum similarity
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.sum(maxsim))

            # check for convergence
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or obj[-1]-obj[-2]<tol:
                    C_final.append(C)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                A = X[idx_k].T@X[idx_k]
                C[:,k] = A@C[:,k]
                # C[:,k] = scipy.sparse.linalg.svds(A,k=1)[2][0] #gives the same result as above but
            C = C/np.linalg.norm(C,axis=0) # normalize the cluster centers
            iter += 1
    try:
        best = np.nanargmax(np.array(obj_final))
    except: 
        if call>4:
            raise ValueError('Diametrical clustering ++ didn''t work for 5 re-calls of the function')
        print('Diametrical clustering returned nan. Repeating')
        return diametrical_clustering(X,K,max_iter=max_iter,num_repl=num_repl,init=init,call=call+1)
    
    return C_final[best],part_final[best],obj_final[best]