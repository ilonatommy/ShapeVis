from __future__ import division

import numpy as np
from source import randomizer
from source.graph import Graph


class WitnessComplexGraphBuilder:
    def __init__(self, original_input, m : int):
        self.original_input = original_input
        self.m = m
        nodes = randomizer.Randomizer(list(self.original_input.data)).sample(self.m)
        self.graph = Graph(nodes)

    def __create_unsampled_nodes(self):
        unsampled_nodes = [node for node in self.original_input.data if not self.graph.has_node(node)]
        return unsampled_nodes
    
    def build_knn(self, k = 1):
        for node1 in self.graph.nodes:
            for node2 in self.graph.nodes:
                if self.graph.are_equal_nodes(node1, node2):
                    continue
                if self.graph.has_node_less_than_k_neighbors(node1, k):
                    self.graph.adjacency_dict[str(node1)].append(node2)
                    self.graph.adjacent_nodes_dist[str(node1)].append(self.graph.calculate_distance(node1, node2))
                else:
                    max_dist = np.max(self.graph.adjacent_nodes_dist[str(node1)])
                    new_node_dist = self.graph.calculate_distance(node1, node2)
                    if new_node_dist < max_dist:
                        index = self.graph.adjacent_nodes_dist[str(node1)].index(max_dist)
                        self.graph.adjacency_dict[str(node1)][index] = node2
                        self.graph.adjacent_nodes_dist[str(node1)][index] = new_node_dist

    def build_augmented_knn(self):
        unsampled_nodes = self.__create_unsampled_nodes()

        for us_node in unsampled_nodes:
            distances = []
            nodes = []
            for node in self.graph.nodes:
                distances.append(self.graph.calculate_distance(node, us_node))
                nodes.append(node)
            # check which 2 nodes in graph are the nearest neighbours
            min_distance1 = min(distances)
            nearest_node1 = nodes[distances.index(min_distance1)]
            distances.remove(min_distance1)
            nodes = [elem for elem in nodes if (list(elem) != nearest_node1).any()]

            min_distance2 = min(distances)
            nearest_node2 = nodes[distances.index(min_distance2)]
            # if these nodes are not adjacent yet, connect them
            if not self.graph.is_node2_neighbor_of_node1(node1=nearest_node1, node2=nearest_node2):
                self.graph.adjacency_dict[str(nearest_node1)].append(nearest_node2)
                dist = self.graph.calculate_distance(nearest_node1, nearest_node2)
                self.graph.adjacent_nodes_dist[str(nearest_node1)] = dist

            if not self.graph.is_node2_neighbor_of_node1(node1=nearest_node2, node2=nearest_node1):
                self.graph.adjacency_dict[str(nearest_node2)].append(nearest_node1)
                dist = self.graph.calculate_distance(nearest_node1, nearest_node2)
                self.graph.adjacent_nodes_dist[str(nearest_node2)] = dist

    def get_graph(self):
        return self.graph
