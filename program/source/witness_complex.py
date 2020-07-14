from __future__ import division
import numpy as np
import pandas as pd
import networkx as nx
from source import randomizer, helpers
from source.helpers import calculate_distance


class WitnessComplexGraphBuilder:
    def __init__(self, original_input, m : int):
        self.original_input = original_input
        indexes = randomizer.Randomizer(list(range(len(self.original_input.data)))).sample(m)

        nodes = self.original_input.data[indexes]
        labels = self.original_input.labels[indexes]
        
        node_names = [str(nd) for nd in nodes]
        
        sampled_graph_weights = np.fromfunction(lambda i, j: (abs(nodes[i] - nodes[j])), shape=(len(nodes), len(nodes)), dtype=int)
        sampled_graph_weights = np.sum(sampled_graph_weights, axis=2)
        adjacency_df = pd.DataFrame(sampled_graph_weights, index=node_names, columns=node_names)
        
        self.sampled_graph = nx.from_pandas_adjacency(adjacency_df)

        nodes_info = map(lambda i: (str(nodes[i]), {"indices": nodes[i], "label": labels[i]}), range(len(nodes)))
        self.knn_graph = nx.Graph()
        self.knn_graph.add_nodes_from(list(nodes_info))

    def __create_unsampled_nodes(self):
        unsampled_nodes = [node for node in self.original_input.data if not self.knn_graph.has_node(node)]
        return unsampled_nodes
    
    def build_knn(self, k=1):
        for node in self.sampled_graph.nodes:
            node_neighbors_edges = sorted(self.sampled_graph.edges(str(node), data=True), key=lambda e: e[2]["weight"])[:k]
            self.knn_graph.add_edges_from(node_neighbors_edges)

    def build_augmented_knn(self):
        unsampled_nodes = self.__create_unsampled_nodes()

        for us_node in unsampled_nodes:
            distances = []
            nodes = []
            for node in self.knn_graph.nodes:
                distances.append(calculate_distance(self.knn_graph.nodes[node]["indices"], us_node))
                nodes.append(node)
            # check if there are at least 2 neighbors that neighborhood could be witnessed
            if (len(distances) < 2):
                continue
            # check which 2 nodes in graph are the nearest neighbours
            min_distance1 = min(distances)
            nearest_node1 = nodes[distances.index(min_distance1)]
            distances.remove(min_distance1)
            nodes = [elem for elem in nodes if (elem != nearest_node1)]

            min_distance2 = min(distances)
            nearest_node2 = nodes[distances.index(min_distance2)]
            # if these nodes are not adjacent yet, connect them
            if not self.knn_graph.has_edge(nearest_node1, nearest_node2):
                self.knn_graph.add_edge(nearest_node1, nearest_node2)

    def get_graph(self):
        return self.knn_graph
