import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import source.helpers as helpers


class CommunityDetector:

    @staticmethod
    def detect_communities(weight_matrix: np.array, labels: list, visualize: bool = True):
        graph = nx.from_numpy_matrix(np.asarray(weight_matrix))
        labels_dict = {i: int(labels[i]) for i in range(len(labels))}
        partition = community_louvain.best_partition(graph)

        if visualize:
            fig = figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
            fig.suptitle("GL graph with communities coloured", fontsize=24)
            pos = nx.spring_layout(graph)
            cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
            nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=150, cmap=cmap, node_color=list(partition.values()))
            nx.draw_networkx_edges(graph, pos, alpha=0.5)
            nx.draw_networkx_labels(graph, pos, labels_dict, font_size=16, font_color='red')
            plt.show()

        labels_per_community = {}
        for node, comm in partition.items():
          if not comm in labels_per_community.keys():
            labels_per_community[comm] = []
          labels_per_community[comm].append(labels_dict[node])

        label_per_community = {}
        for comm in labels_per_community.keys():
          label_per_community[comm] = (helpers.most_common(labels_per_community[comm]))

        induced_graph = community_louvain.induced_graph(partition, graph)

        if visualize:
          helpers.visualize_graph(induced_graph, label_per_community, "IGP graph - induced after community detection")

        return induced_graph, list(label_per_community.values())