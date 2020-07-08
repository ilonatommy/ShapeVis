from __future__ import division

import numpy as np
from source import randomizer
import networkx as nx


class WitnessComplexGraphBuilder:
    def __init__(self, original_input, m : int):
        self.original_input = original_input
        indexes = randomizer.Randomizer(list(range(len(self.original_input.data)))).sample(m)
        nodes = self.original_input.data[indexes]
        labels = self.original_input.labels[indexes]
        
        self.G = nx.Graph()
        # TODO consider removing indices
        nds = map(lambda i: (str(nodes[i]), {"indices":nodes[i], "label": labels[i]}), range(len(nodes)))
        
        self.G.add_nodes_from(list(nds))

    def __create_unsampled_nodes(self):
        unsampled_nodes = [node for node in self.original_input.data if not self.G.has_node(node)]
        return unsampled_nodes
    
    def build_knn(self, k = 1):
        # TODO consider avoiding graph copy
        graph_cpy = self.G.copy()
        for i in graph_cpy.nodes:
            for j in graph_cpy.nodes:
                if i != j:
                    graph_cpy.add_edge(i, j, weight=self.calculate_distance(graph_cpy.nodes[i]["indices"], graph_cpy.nodes[j]["indices"]))
        
        for node in graph_cpy.nodes:
            node_neighbors_edges = sorted(graph_cpy.edges(node, data=True), key=lambda e: e[2]["weight"])[:k]
            self.G.add_edges_from(node_neighbors_edges)

    # TODO maybe the algorithm could be simplified
    def build_augmented_knn(self):
        unsampled_nodes = self.__create_unsampled_nodes()

        for us_node in unsampled_nodes:
            distances = []
            nodes = []
            for node in self.G.nodes:
                distances.append(self.calculate_distance(self.G.nodes[node]["indices"], us_node))
                nodes.append(node)
            # check which 2 nodes in graph are the nearest neighbours
            min_distance1 = min(distances)
            nearest_node1 = nodes[distances.index(min_distance1)]
            distances.remove(min_distance1)
            nodes = [elem for elem in nodes if (elem != nearest_node1)]

            min_distance2 = min(distances)
            nearest_node2 = nodes[distances.index(min_distance2)]
            # if these nodes are not adjacent yet, connect them
            if not self.G.has_edge(nearest_node1, nearest_node2):
                self.G.add_edge(nearest_node1, nearest_node2)

    @staticmethod
    def calculate_distance(node1, node2):
        if node1.shape != node2.shape:
            return -1
        sum_dist = 0
        for dim in range(node1.shape[0]):
            sum_dist += (node1[dim] - node2[dim]) * (node1[dim] - node2[dim])
        return np.sqrt(sum_dist)

    def get_graph(self):
        return self.G
