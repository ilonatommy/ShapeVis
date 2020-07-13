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

        modularity_without_edge = {}
        for edge in igp_graph.edges:
            graph_cpy = igp_graph.copy()
            graph_cpy.remove_edge(edge[0], edge[1])
            modularity = helpers.calculate_modularity(graph_cpy)
            modularity_without_edge[edge] = modularity

        result_graph = nx.Graph()
        result_graph.add_nodes_from(igp_graph.nodes)

        prev_modularity = MINIMAL_MODULARITY
        for edge, _ in sorted(modularity_without_edge.items(), key=lambda item: item[1]):
            result_graph.add_edge(edge[0], edge[1])
            modularity = helpers.calculate_modularity(result_graph)
            if nx.number_connected_components(result_graph) == initial_no_of_components:
                break
            if prev_modularity > modularity:
                result_graph.remove_edge(edge[0], edge[1])
                break
            prev_modularity = helpers.calculate_modularity(result_graph)

        if visualize:
            helpers.visualize_graph(result_graph, labels_dict, "Final graph - after manifold tearing")

        return result_graph
