from sklearn import cluster
import numpy
def spectral(dataset, n_cluster = 625):
    import util.affinity_matrix
    feature_matrix = numpy.asarray(dataset)
    mode = "GPUAffinity"
    if mode == "GPUAffinity":
        print "Calculate affinity."
        affinty_matrix = util.affinity_matrix.compute_affinity_gaussian_matrix(feature_matrix)[0]
        print "Calculated affinity. Start to cluster."
        
        spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                          eigen_solver='arpack',
                                          affinity="precomputed")
        spectral.fit(affinty_matrix)
    else:
        spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
        spectral.fit(dataset)
    cluster_labels = spectral.labels_.astype(numpy.int)
    print "Clustering finished."
    return cluster_labels

def kmeans(dataset, n_cluster = 625):
    from scipy.cluster.vq import kmeans2, whiten
    feature_matrix = numpy.asarray(dataset)
    whitened = whiten(feature_matrix)
    cluster_num = 625
    _, cluster_labels = kmeans2(whitened, cluster_num)
    return cluster_labels