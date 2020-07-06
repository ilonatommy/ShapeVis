import numpy as np


class Graph:
    def __init__(self, nodes, stringified_hash=True):
        self.nodes = nodes
        hashable_nodes = []
        if stringified_hash:
            for node in nodes:
                hashable_nodes.append(str(node))
        else:
            hashable_nodes = nodes
        self.adjacency_dict = {i: [] for i in hashable_nodes}
        self.adjacent_nodes_dist = {i: [] for i in hashable_nodes}

    # TODO zmienić docelowo na are_neighobours czy coś takiego być może - to graf bezkierunkowy
    def is_node2_neighbor_of_node1(self, node1, node2, stringified_hash=True):
        if stringified_hash:
            for n in self.adjacency_dict[str(node1)]:
                if self.are_equal_nodes(n, node2):
                    return True
            return False
        else:
            for n in self.adjacency_dict[node1]:
                if self.are_equal_nodes(n, node2, stringified_hash):
                    return True
            return False

    def has_node_less_than_k_neighbors(self, node, k):
        return len(self.adjacency_dict[str(node)]) < k

    def has_node(self, node):
        for n in self.nodes:
            if self.are_equal_nodes(n, node):
                return True
        return False

    # TODO jakby inaczej przechowywać to, można by uniknąć przechodzenia po całym słowniku
    # żeby wychwycić węzeł w jakiejś liście
    def __remove_node_from_neigbours_list(self, node: list):
        for key in self.adjacency_dict:
            self.adjacency_dict[key] = [n for n in self.adjacency_dict[key] if (list(n) != node).any()]

    def __remove_neigbours_list_of_node(self, node: list):
        del self.adjacency_dict[str(node)]

    def __remove_node(self, node):
        self.nodes = [n for n in self.nodes if (list(n) != node).any()]

    def remove_node(self, node: list):
        self.__remove_node_from_neigbours_list(node)
        self.__remove_neigbours_list_of_node(node)
        self.__remove_node(node)

    @staticmethod
    def are_equal_nodes(node1, node2, stringified_hash=True):
        if stringified_hash:
            return node1.shape == node2.shape and (list(node1) == node2).all()
        else:
            return node1 == node2

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
