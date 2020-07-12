import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from source.helpers import most_common


class CommunityDetector:

    @staticmethod
    def detect_communities(weight_matrix: np.array, labels: list, visualize: bool = False):
        graph = nx.from_numpy_matrix(np.asarray(weight_matrix))
        labels_dict = {i: labels[i] for i in range(len(labels))}
        partition = community_louvain.best_partition(graph)

        if visualize:
            figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
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
          label_per_community[comm] = (most_common(labels_per_community[comm]))
        print(labels_per_community)
        print(label_per_community)

        induced_graph = community_louvain.induced_graph(partition, graph)

        if visualize:
            figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
            pos=nx.spring_layout(induced_graph)
            nx.draw_networkx_nodes(induced_graph, pos, node_color=list(label_per_community.values()))
            nx.draw_networkx_edges(induced_graph, pos, alpha=0.5)
            nx.draw_networkx_labels(induced_graph, pos, label_per_community, font_size=16, font_color='red')
            plt.show()

        return induced_graph