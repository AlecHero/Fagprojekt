from clustering_basic import kmeans_clustering, diametrical_clustering
from clustering_grassmannian import grassmannian_clustering_chordal, weighted_grassmann_clustering
from clustering_grassmannian import grassmannian_clustering_geodesic
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import os

def get_save_path(method, gen, N): return f"generated_data\clustering_{method}{'_g' if gen else ''}_N{N}.pkl"
def load_clust(method="kc", gen=False, N=20): return pickle.load(open(get_save_path(method,gen,N), "rb"))
def save_clust(clust_dict, method="kc", gen=False, N=20): return pickle.dump(clust_dict, open(get_save_path(method,gen,N), "wb"))

def get_true_labels(K, N=10_000): return np.pad(np.repeat(np.arange(K), N//K), (0, N%K), mode='constant', constant_values=K)
def get_nmi_dict(c_data): return {(p,K) : [calc_NMI(labels, get_true_labels(K)) for (_, labels, _) in data] for (p,K), data in c_data.items()}
def coh_map(x): return np.outer(np.cos(x), np.cos(x)) + np.outer(np.sin(x), np.sin(x))

def _NMI_convert(labels): return np.array([(labels == val).astype(int) for val in np.unique(labels)])

def calc_MI(Z1, Z2):
    if isinstance(Z1, (list, np.ndarray)) and not isinstance(Z1[0], (list, np.ndarray)): Z1 = _NMI_convert(Z1)
    if isinstance(Z2, (list, np.ndarray)) and not isinstance(Z2[0], (list, np.ndarray)): Z2 = _NMI_convert(Z2)
    # Z1 and Z2 are two partition matrices of size (KxN) where K is number of components and N is number of samples
    P = Z1 @ Z2.T # joint probability matrix
    PXY = P / np.sum(P) # joint probability matrix normalized
    PXPY = np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0)) # product of marginals
    ind = np.where(PXY > 0) # non-zero elements
    MI = np.sum(PXY[ind] * np.log(PXY[ind]/PXPY[ind])) # mutual information
    return MI

def calc_NMI(Z1, Z2): return calc_MI(Z1,Z2) / np.mean([calc_MI(Z1,Z1), calc_MI(Z2,Z2)])


def get_data(p, K):
    synth_data = np.loadtxt('synthetic_isotropic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
    n = synth_data.shape[0]
    data_gr = np.zeros((int(n/2),p,2))
    data_gr[:,:,0] = synth_data[np.arange(n,step=2),:] # first frame
    data_gr[:,:,1] = synth_data[np.arange(n,step=2)+1,:] # second frame
    data_sphere = data_gr[:,:,0]
    return data_sphere, data_gr


def generate_data(p, K, N=10_000, noise_scale=0.7, seed=None):
    if seed is not None: np.random.seed(seed)
    
    noise = np.random.rand(p,N) * noise_scale
    data_gr = np.zeros((N,p,2))
    data_eigvals = np.zeros((N,2))
    theta_bases = [] # cluster_base_vectors

    for i in (np.linspace(0,p-1,K, dtype=int)):
        theta_base = np.ones(p)*np.pi/2
        theta_base[i] = 0
        theta_bases.append(theta_base)

    for j in range(K):
        theta_cluster = np.asarray([theta_bases[j] for _ in range(N)]) + noise.T
        for i in range(N//K):
            eigvals,eigvecs = np.linalg.eigh(coh_map(theta_cluster[i]))
            eigvals,eigvecs = eigvals[-2:],eigvecs[:,-2:]
            data_gr[i+j*N//K,:,0] = eigvecs[:,np.argmax(eigvals)]
            data_gr[i+j*N//K,:,1] = eigvecs[:,np.argmin(eigvals)]
            data_eigvals[i+j*N//K,0] = eigvals[np.argmax(eigvals)]
            data_eigvals[i+j*N//K,1] = eigvals[np.argmin(eigvals)]
    data_sphere = data_gr[:,:,0]
    return data_sphere, data_gr, data_eigvals


def single_clustering(method="kc", N=25, force=False, generated_data=False):
    pK_comb = list(zip([3,10,10,25,25,25,50,50,50,100,100,100,500,500,500],[2,2,5,2,5,10,2,5,10,2,5,10,2,5,10]))

    if force or not os.path.exists(get_save_path(method, generated_data, N)):
        result_dict = {}
        for p,K in tqdm(pK_comb, total=len(pK_comb)):
            if generated_data: data_sphere, data_gr, data_eigvals = generate_data(p, K)
            else: data_sphere, data_gr = get_data(p, K)
            raw_df = []
            for _ in tqdm(range(N), leave=False):
                if   method == "kc":  raw_df.append(kmeans_clustering(data_sphere, K))
                elif method == "dc":  raw_df.append(diametrical_clustering(data_sphere, K))
                elif method == "gcc": raw_df.append(grassmannian_clustering_chordal(data_gr, K))
                elif method == "ggc": raw_df.append(grassmannian_clustering_geodesic(data_gr, K))
                elif method == "wgc": raw_df.append(weighted_grassmann_clustering(data_gr, data_eigvals, K))
            result_dict[(p,K)] = raw_df
        with open(get_save_path(method, generated_data, N), "wb") as file: pickle.dump(result_dict, file)
    else:
        result_dict = pickle.load(open(get_save_path(method, generated_data, N), "rb"))
    return result_dict


def plot_sphere(data_sphere, centroids, label):
    import plotly.express as px
    px.defaults.color_continuous_scale = px.colors.sequential.Sunset
    
    if label is not None: fig = px.scatter_3d(x=data_sphere[:,0], y=data_sphere[:,1], z=data_sphere[:,2], color=label, opacity=0.5, template="plotly_dark")
    else: fig = px.scatter_3d(x=data_sphere[:,0], y=data_sphere[:,1], z=data_sphere[:,2], opacity=0.5, template="plotly_dark")
    fig.update_traces(marker_size = 2)
    
    if centroids is not None:
        try: fig.add_scatter3d(x=centroids[:,0], y=centroids[:,1], z=centroids[:,2], mode='markers', marker=dict(size=8, color='red'))
        except: print("Could not plot centroids")
    fig.show()


if __name__ == "__main__":
    

    p,K,trial,method = 3,2,0,"gcc"
    data_sphere, data_gr = get_data(p,K)
    clustering_dict = load_clust(method)
    centroids, labels, obj = np.array(clustering_dict[p,K], dtype=object).T

    plot_sphere(data_sphere, centroids[trial], labels[trial])