import numpy as np

def calculate_distance(node1, node2):
    if node1.shape != node2.shape:
        return -1
    sum_dist = 0
    for dim in range(node1.shape[0]):
        sum_dist += (node1[dim] - node2[dim]) * (node1[dim] - node2[dim])
    return np.sqrt(sum_dist)

def most_common(lst):
    return max(set(lst), key=lst.count)
