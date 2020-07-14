import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import source.helpers as helpers

MINIMAL_MODULARITY = -0.5

class ManifoldTearer:
    @staticmethod
    def reduce_edges(igp_graph: nx.Graph, labels: list, visualize: bool = True):
        initial_no_of_components = nx.number_connected_components(igp_graph)

        if visualize:
            labels_dict = {i: int(labels[i]) for i in range(len(labels))}
            helpers.visualize_graph(igp_graph, labels_dict, "IGP graph - before manifold tearing")

        igp_modularity = helpers.calculate_modularity(igp_graph)
        modularity_heap = {}
        for edge in igp_graph.edges:
            graph_cpy = igp_graph.copy()
            graph_cpy.remove_edge(edge[0], edge[1])
            modularity = helpers.calculate_modularity(graph_cpy)
            modularity_heap[edge] = igp_modularity - modularity

        result_graph = nx.Graph()
        result_graph.add_nodes_from(igp_graph.nodes)

        sorted_mod_heap = (sorted(modularity_heap.items(), key=lambda item: item[1]))[::-1]
        for edge, edge_modularity in sorted_mod_heap:
            if nx.number_connected_components(result_graph) == initial_no_of_components:
                break
            result_graph.add_edge(edge[0], edge[1])


        if visualize:
            helpers.visualize_graph(result_graph, labels_dict, "Final graph - after manifold tearing")

        return result_graph
