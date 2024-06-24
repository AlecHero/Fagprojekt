import numpy as np

def _NMI_convert(labels): return np.array([(labels == val).astype(int) for val in np.unique(labels)])

def calc_MI(Z1, Z2):
    if isinstance(Z1, (list, np.ndarray)) and not isinstance(Z1[0], (list, np.ndarray)): Z1 = _NMI_convert(Z1)
    if isinstance(Z2, (list, np.ndarray)) and not isinstance(Z2[0], (list, np.ndarray)): Z2 = _NMI_convert(Z2)
    
    P = Z1 @ Z2.T # joint probability matrix
    PXY = P / np.sum(P) # joint probability matrix normalized
    PXPY = np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0)) # product of marginals
    ind = np.where(PXY > 0) # non-zero elements
    MI = np.sum(PXY[ind] * np.log(PXY[ind]/PXPY[ind])) # mutual information
    return MI

def calc_NMI(Z1, Z2):
    if isinstance(Z1, (list, np.ndarray)) and not isinstance(Z1[0], (list, np.ndarray)): Z1 = _NMI_convert(Z1)
    if isinstance(Z2, (list, np.ndarray)) and not isinstance(Z2[0], (list, np.ndarray)): Z2 = _NMI_convert(Z2)
    
    # Z1 and Z2 are two partition matrices of size (KxN) where K is number of components and N is number of samples
    NMI = calc_MI(Z1,Z2) / np.mean([calc_MI(Z1,Z1), calc_MI(Z2,Z2)])
    return NMI

