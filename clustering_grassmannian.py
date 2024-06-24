import numpy as np
import scipy


def plusplus_initialization(X,K,seed=None,X_weights=None,dist='diametrical'):
    if seed is not None: np.random.seed(seed)
    
    assert dist in ['diametrical','grassmann','weighted_grassmann']
    n = X.shape[0]
    if X.ndim == 3:
        q = X.shape[2]

    # choose first centroid at random from X
    idx = np.random.choice(n,p=None)
    
    X = np.delete(X,idx,axis=0)

    if dist == 'diametrical':
        C = X[idx][:,np.newaxis]
    else:
        C = np.zeros((K,X.shape[1],X.shape[2]))
        C[0] = X[idx]
        if X_weights is not None:
            C_weights = np.zeros((K,X.shape[2]))
            C_weights[0] = X_weights[idx]
            X_weights = np.delete(X_weights,idx,axis=0)

    # for all other centroids, compute the distance from all X to the current set of centroids. 
    # Construct a weighted probability distribution and sample using this. 

    for k in range(K):
        if dist == 'diametrical':
            dis = 1-(X@C)**2 
            dis = np.clip(dis,0,None)
        elif dist == 'grassmann':
            dis = 1/np.sqrt(2)*(2*q-2*np.linalg.norm(np.swapaxes(X[:,None],-2,-1)@C[:k+1][None],axis=(-2,-1)))
            dis = np.clip(dis,0,None)
        elif dist == 'weighted_grassmann':
            dis = 1/np.sqrt(2)*(np.sum(X_weights**4,axis=1)[:,None]+np.sum(C_weights**4,axis=1)[None]-2*np.linalg.norm(np.swapaxes((X*X_weights[:,None,:])[:,None],-2,-1)@(C*C_weights[:,None,:])[None],axis=(-2,-1)))
            # dis = 1/np.sqrt(2)(np.sum(L**4)+np.sum(-2*np.linalg.norm(np.swapaxes((X*L[:,None,:])[:,None],-2,-1)@C[:k+1][None],axis=(-2,-1)))

        mindis = np.min(dis,axis=1) #choose the distance to the closest centroid for each point

        if k==K-1:
            X_part = np.argmin(dis,axis=1)
            obj = np.mean(mindis)
            break

        prob_dist = mindis/np.sum(mindis) # construct the prob. distribution
        prob_dist = abs(prob_dist) # avoid numerical issues (negative probabilities due to numerical errors
        prob_dist = prob_dist/np.sum(prob_dist) # normalize the distribution
        idx = np.random.choice(n-k-1,p=prob_dist)
        if dist == 'diametrical':
            C = np.hstack((C,X[idx][:,np.newaxis]))
        else:
            C[k+1] = X[idx]
        X = np.delete(X,idx,axis=0)
        if X_weights is not None:
            C_weights[k+1] = X_weights[idx]
            X_weights = np.delete(X_weights,idx,axis=0)
    if X_weights is not None:
        return C,C_weights,X_part,obj
    else:
        return C,X_part,obj


def grassmannian_clustering_chordal(X,K,seed=None,max_iter=10000,tol=1e-16):
    if seed is not None: np.random.seed(seed)
    """"
    Implementation of Grassmann clustering as in Gruber & Theis 2006 https://ieeexplore.ieee.org/abstract/document/7071681

    X: size (nxpxq), where n is the number of observations, p is the number of features and q is the subspace dimensionality
    K: number of clusters
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    
    returns: cluster centers, objective function values, and cluster assignments
    """
    
    n,p,q = X.shape

    # initialize cluster centers using a normal distribution projected to the Grassmannian
    C = np.random.randn(K,p,q)
    for k in range(K):
        C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

    # initialize counters
    iter = 0
    obj = []
    partsum = np.zeros((max_iter,K))
    while True:

        # "E-step" - compute the similarity between each matrix and each cluster center
        dis = 1/np.sqrt(2)*(2*X.shape[-1]-2*np.linalg.norm(np.swapaxes(X[:,None],-2,-1)@C[None],axis=(-2,-1)))
        sim = -dis
        maxsim = np.max(sim,axis=1) # find the maximum similarity - the sum of this value is the objective function
        X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        obj.append(np.sum(maxsim))

        # check for convergence
        for k in range(K):
            partsum[iter,k] = np.sum(X_part==k)
        if iter>0:
            if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                break
        
        # "M-step" - update the cluster centers
        for k in range(K):
            idx_k = X_part==k
            V = np.sum(X[idx_k]@np.swapaxes(X[idx_k],1,2),axis=0) #extrinsic manifold mean, may be computed more efficiently using the SVD
            L,U = scipy.sparse.linalg.eigsh(V,k=q,which='LM')
            C[k] = U
            
        iter += 1
    
    return C,X_part,obj


def grassmannian_clustering_geodesic(X,K,max_iter=10000,tol=1e-16):
    """"
    Implementation of Grassmann clustering with a geodesic distance function and a SVD-based update rule

    X: size (nxpxq), where n is the number of observations, p is the number of features and q is the subspace dimensionality
    K: number of clusters
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    
    returns: cluster centers, objective function values, and cluster assignments
    """

    
    n,p,q = X.shape
    
    # initialize cluster centers using a normal distribution projected to the Grassmannian
    C = np.random.randn(K,p,q)
    for k in range(K):
        C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian

    # initialize counters
    iter = 0
    obj = []
    partsum = np.zeros((max_iter,K))
    while True:

        # "E-step" - compute the similarity between each matrix and each cluster center
        _,S,_ = np.linalg.svd(np.swapaxes(X[:,None],-2,-1)@C[None],full_matrices=False)
        dis = np.linalg.norm(np.arccos(np.clip(S,-1,1)),axis=-1)
        sim = -dis
        maxsim = np.max(sim,axis=1) # find the maximum similarity - the sum of this value is the objective function
        X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
        obj.append(np.sum(maxsim))

        # check for convergence
        for k in range(K):
            partsum[iter,k] = np.sum(X_part==k)
        if iter>0:
            if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                break
        
        # "M-step" - update the cluster centers
        for k in range(K):
            idx_k = X_part==k
            V = np.reshape(np.swapaxes(X[idx_k],0,1),(p,np.sum(idx_k)*q))
            U,_,_ = scipy.sparse.linalg.svds(V,q)
            C[k] = U[:,:q]

        iter += 1
    
    return C,X_part,max(obj)


def weighted_grassmann_clustering(X,X_weights,K,seed=None,max_iter=10000,num_repl=1,tol=1e-16,init=None):
    """"
    Weighted grassmannian clustering using the chordal distance function and a SVD-based update rule
    
    X: size (nxpxq), where n is the number of observations, p is the number of features and q is the subspace dimensionality
    X_weights: size (n,q), where n is the number of observations and q is the subspace dimensionality (corresponds to eigenvalues)
    K: number of clusters
    max_iter: maximum number of iterations
    tol: tolerance for convergence
 
    """
    
    n,p,q = X.shape
 
    obj_final = [] # objective function collector
    part_final = [] # partition collector
    C_final = [] # cluster center collector
    C_weights_final = [] # cluster weights collector
 
    # loop over the number of repetitions
    for _ in range(num_repl):
        # initialize cluster centers
        if init is None or init=='++' or init=='plusplus' or init == 'weighted_grassmann_clustering_plusplus':
            C,C_weights,_,_ = plusplus_initialization(X,K,seed=seed,X_weights=X_weights,dist='grassmann')
        elif init=='uniform' or init=='unif':
            C = np.random.uniform(size=(K,p,q))
            C_weights = np.ones((K,q))
            for k in range(K):
                C[k] = C[k]@scipy.linalg.sqrtm(np.linalg.inv(C[k].T@C[k])) # project onto the Grassmannian
 
        # initialize counters
        iter = 0
        obj = []
        partsum = np.zeros((max_iter,K))
        while True:
            # "E-step" - compute the similarity between each matrix and each cluster center
            M = np.swapaxes(X*np.sqrt(X_weights[:,None,:]),-2,-1)[:,None]@((C*np.sqrt(C_weights)[:,None,:])[None])
            dis = 1/np.sqrt(2)*(np.sum(X_weights**2,axis=1)[:,None]+np.sum(C_weights**2,axis=1)[None]-2*np.linalg.norm(M,axis=(-2,-1))**2)#
            sim = -dis
            maxsim = np.max(sim,axis=1) # find the maximum similarity - the sum of this value is the objective function
            X_part = np.argmax(sim,axis=1) # assign each point to the cluster with the highest similarity
            obj.append(np.sum(maxsim))
 
            # check for convergence   
            for k in range(K):
                partsum[iter,k] = np.sum(X_part==k)
            if iter>0:
                if all((partsum[iter-1]-partsum[iter])==0) or iter==max_iter or abs(obj[-1]-obj[-2])<tol:
                    C_final.append(C)
                    C_weights_final.append(C_weights)
                    obj_final.append(obj[-1])
                    part_final.append(X_part)             
                    break
            
            # "M-step" - update the cluster centers
            for k in range(K):
                idx_k = X_part==k
                V = np.reshape(np.swapaxes(X[idx_k]*X_weights[idx_k,None,:],0,1),(p,np.sum(idx_k)*q))
                U,S,_ = scipy.sparse.linalg.svds(V,q)
                C[k] = U#[:,::-1]
                # C[k] = U[:,:q]
                C_weights[k] = S#[::-1]
                C_weights[k] = C_weights[k]/np.sum(C_weights[k])*p
 
            iter += 1
    best = np.nanargmax(np.array(obj_final))
    return C_final[best],C_weights_final[best],part_final[best],obj_final[best]