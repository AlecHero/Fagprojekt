import numpy as np
from tqdm import tqdm
from riemannian_kmeans import kmeans_clustering, cosine_clustering, diametrical_clustering
from riemannian_kmeans import grassmannian_clustering_gruber2006, weighted_grassmannian_clustering

def clustering(K, p, N):
    K = 2; p = 3
    centroids = np.loadtxt('synthetic_isotropic/centroids/synth_cov_p'+str(p)+'K'+str(K)+'_'+str(1)+'.csv',delimiter=',')
    synth_data = np.loadtxt('synthetic_isotropic/synth_data_MACG_p'+str(p)+'K'+str(K)+'_1.csv',delimiter=',')
    n = synth_data.shape[0]

    data_gr = np.zeros((int(n/2),p,2))
    data_gr[:,:,0] = synth_data[np.arange(n,step=2),:] # first frame
    data_gr[:,:,1] = synth_data[np.arange(n,step=2)+1,:] # second frame
    data_sphere = data_gr[:,:,0]

    labels_list = []
    centroids = []
    for _ in tqdm(range(N)):
        row = []
        row.append(kmeans_clustering(data_sphere, K))
        row.append(cosine_clustering(data_sphere, K))
        row.append(diametrical_clustering(data_sphere, K))
        row.append(grassmannian_clustering_gruber2006(data_gr, K))
        row.append(weighted_grassmannian_clustering(data_gr, K, max_iter=100))
        
        centroids.append([row[0][0], row[1][0], row[2][0].T, row[3][0].transpose((0,2,1)), row[4][0].transpose((0,2,1))])
        labels_list.append([i[1] for i in row])

    np.save(f'centroids_N{N}.npy', np.array(centroids, dtype=object))
    np.save(f'clustering_labels_N{N}.npy', labels_list)
    return centroids, labels_list