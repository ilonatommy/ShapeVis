from __future__ import division
import numpy as np


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        hashable_nodes = []
        for node in nodes:
            hashable_nodes.append(str(node))
        self.adjacency_dict = {i: [] for i in hashable_nodes}
        self.adjacent_nodes_dist = {i: [] for i in hashable_nodes}

    def compare_nodes(self, node1, node2):
        if node1.shape != node2.shape:
            return False
        for dim in range(len(node1.shape)):
            if node1[dim] != node2[dim]:
                return False
        return True

    def calculate_distance(self, node1, node2):
        if node1.shape != node2.shape:
            return -1
        sum_dist = 0
        for dim in range(len(node1.shape)):
            sum_dist += (node1[dim] - node2[dim]) * (node1[dim] - node2[dim])
        return np.sqrt(sum_dist)


class ManifoldLandmarker:
    def __init__(self, data_proc, distance):
        self.mins =  np.amin(data_proc.data, axis=0)# min values for consecutive dimensions: 0, 1..
        self.distance = distance

    def __sample(self, points, dim, element_dims, results):
        if len(points.shape) > 1:
            for i in range(points.shape[dim]):
                self.__sample(points[i], dim+1, element_dims, results)
        else:
            sample = True
            for point in points:
                sampled_val = (point - self.mins[dim]) / self.distance
                if int(sampled_val) != sampled_val:
                    sample = False
                    break
            if sample:
                results.append(points)
        return results

    def __uniform_sampling(self, data_proc):
        nodes = []
        return self.__sample(data_proc.data, 0, len(data_proc.data.shape), nodes)

    def __define_k_nn(self, k, graph):
        for node1 in graph.nodes:
            nn = 0
            for node2 in graph.nodes:
                if graph.compare_nodes(node1, node2):
                    continue
                if len(graph.adjacency_dict[str(node1)]) < k:
                    graph.adjacency_dict[str(node1)].append(node2)
                    graph.adjacent_nodes_dist[str(node1)].append(graph.calculate_distance(node1, node2))
                else:
                    max_dist = np.max(graph.adjacent_nodes_dist[str(node1)])
                    new_node_dist = graph.calculate_distance(node1, node2)
                    if new_node_dist < max_dist:
                        index = graph.adjacent_nodes_dist[str(node1)].index(max_dist)
                        graph.adjacency_dict[str(node1)][index] = node2
                        graph.adjacent_nodes_dist[str(node1)][index] = new_node_dist


    def create_knn_graph(self, data_proc):
        k = 1
        nodes = self.__uniform_sampling(data_proc)
        graph = Graph(nodes)
        self.__define_k_nn(k, graph)
        print(graph.adjacency_dict)
