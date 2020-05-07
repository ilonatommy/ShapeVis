import numpy as np


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        hashable_nodes = []
        for node in nodes:
            hashable_nodes.append(str(node))
        self.adjacency_dict = {i: [] for i in hashable_nodes}
        self.adjacent_nodes_dist = {i: [] for i in hashable_nodes}

    # TODO zmienić docelowo na are_neighobours czy coś takiego być może - to graf bezkierunkowy
    def is_node2_neighbor_of_node1(self, node1, node2):
        for n in self.adjacency_dict[str(node1)]:
            if self.are_equal_nodes(n, node2):
                return True
        return False

    def has_node_less_than_k_neighbors(self, node, k):
        return len(self.adjacency_dict[str(node)]) < k

    def has_node(self, node):
        for n in self.nodes:
            if self.are_equal_nodes(n, node):
                return True
        return False

    @staticmethod
    def are_equal_nodes(node1, node2):
        return node1.shape == node2.shape and (list(node1) == node2).all()

    @staticmethod
    def calculate_distance(node1, node2):
        if node1.shape != node2.shape:
            return -1
        sum_dist = 0
        for dim in range(node1.shape[0]):
            sum_dist += (node1[dim] - node2[dim]) * (node1[dim] - node2[dim])
        return np.sqrt(sum_dist)

    def __str__(self):
        graph_summary = ""
        graph_summary += "Graph nodes: " + str(self.nodes)
        graph_summary += "\nGraph adjacency_dict: " + str(self.adjacency_dict)
        graph_summary += "\nGraph adjacent_nodes_dist: " + str(self.adjacent_nodes_dist)
        return graph_summary