import string
import util.cluster as cluster
from sklearn import metrics
import numpy as np

if __name__ == '__main__':
    print "Start!"
    matrix = list()
    ids = list()
    
    dimension = 9500
    zeros = [0.0] * dimension
    
    n_cluster = 2
    f = open("../data/kefu/official_method/small")
    count = 0
    for line in f:
        tokens = line.strip().split("\t")
        line_id = string.atoi(tokens[0])
        ids.append(line_id)
        
        v = map(lambda s: string.atof(s), tokens[1:])
        matrix.append(zeros[:count] + v)
        count += 1
    f.close()
        
    matrix = np.asarray(matrix)
    matrix += np.transpose(matrix)
    
    for i in xrange(len(matrix)):
        matrix[i][i] = 1
    
    cluster_ids = cluster.spectral(matrix, n_cluster=n_cluster, matrix_type="affinity")
    
    base_labels = list()
    f = open("../data/measure/base")
    for line in f:
        tokens = line.split("\t")
        base_labels.append(string.atoi(tokens[1]))
    f.close()
    
    e = metrics.adjusted_mutual_info_score( [base_labels[i] for i in ids], cluster_ids)
    print e
    print "All finished!"