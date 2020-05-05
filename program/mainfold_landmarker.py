from __future__ import division

from random import randrange

import numpy as np
import random


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        hashable_nodes = []
        for node in nodes:
            hashable_nodes.append(str(node))
        self.adjacency_dict = {i: [] for i in hashable_nodes}
        self.adjacent_nodes_dist = {i: [] for i in hashable_nodes}

    def contains_node(self, nodes_list, node):
        contains = False
        for n in nodes_list:
            if self.are_equal_nodes(n, node):
                contains = True
                break
        return contains

    def are_equal_nodes(self, node1, node2):
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


class UniformSampler:
    def __init__(self, data : list):
        self.data = data

    def sample(self, n):
        return random.sample(self.data, n)

class MainfoldLandmarker:
    def __init__(self, original_input, m : int):
        self.original_input = original_input
        self.m = m

    def __define_k_nn(self, k, graph):
        for node1 in graph.nodes:
            nn = 0
            for node2 in graph.nodes:
                if graph.are_equal_nodes(node1, node2):
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

    def __augment_knn(self, data, graph):
        # create unsampled_nodes = set(data) - set(graph.nodes)
        unsampled_nodes = []
        for node1 in data:
            copy = True
            for node2 in graph.nodes:
                if graph.are_equal_nodes(node1, node2):
                    copy = False
                    break
            if copy:
                unsampled_nodes.append(node1)

        for us_node in unsampled_nodes:
            distances = []
            nodes = []
            for node in graph.nodes:
                distances.append(graph.calculate_distance(node, us_node))
                nodes.append(node)
            # check which 2 nodes in graph are the nearest neighbours
            min_distance1 = min(distances)
            nearest_node1 = nodes[distances.index(min_distance1)]
            distances.remove(min_distance1)
            nodes = [elem for elem in nodes if (list(elem) != nearest_node1).all()]

            min_distance2 = min(distances)
            nearest_node2 = nodes[distances.index(min_distance2)]
            # if these nodes are not adjacent yet, connect them
            if not graph.contains_node(graph.adjacency_dict[str(nearest_node1)], nearest_node2):
                graph.adjacency_dict[str(nearest_node1)].append(nearest_node2)
                dist = graph.calculate_distance(nearest_node1, nearest_node2)
                graph.adjacent_nodes_dist[str(nearest_node1)] = dist

            if not graph.contains_node(graph.adjacency_dict[str(nearest_node2)], nearest_node1):
                graph.adjacency_dict[str(nearest_node2)].append(nearest_node1)
                dist = graph.calculate_distance(nearest_node1, nearest_node2)
                graph.adjacent_nodes_dist[str(nearest_node2)] = dist


    def create_knn_graph(self, k = 1):
        nodes = UniformSampler(list(self.original_input.data)).sample(self.m)
        print(nodes)
        graph = Graph(nodes)
        self.__define_k_nn(k, graph)
        self.__augment_knn(self.original_input.data, graph)
        # print(graph.adjacency_dict)
