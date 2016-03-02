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
    with open("../data/kefu/official_method/t6_output") as f:
        count = 0
        for line in f:
            tokens = line.strip().split("\t")
            line_id = string.atoi(tokens[0])
            ids.append(line_id)
            
            v = map(lambda s: string.atof(s), tokens[1:])
            matrix.append(zeros[:count] + v)
            count += 1
        
    matrix = np.asarray(matrix)
    matrix += np.transpose(matrix)
    
    for i in xrange(len(matrix)):
        matrix[i][i] = 1
    
    print "Start to cluster!"
    print "Total data amount: %d" % len(matrix)
    cluster_ids = cluster.spectral(matrix, n_cluster=n_cluster, matrix_type="affinity")
    print "Finish to cluster!"

    base_labels = list()
    with open("../data/measure/base") as f:
        for line in f:
            tokens = line.split("\t")
            base_labels.append(string.atoi(tokens[1]))
    print "Total  base data amount: %d" % len(base_labels)
    
    e = metrics.adjusted_mutual_info_score( [base_labels[i] for i in ids], cluster_ids)
    print e
    print "All finished!"