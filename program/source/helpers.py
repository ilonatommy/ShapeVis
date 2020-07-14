import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def calculate_distance(node1, node2):
    if node1.shape != node2.shape:
        return -1
    sum_dist = 0
    for dim in range(node1.shape[0]):
        sum_dist += (node1[dim] - node2[dim]) * (node1[dim] - node2[dim])
    return np.sqrt(sum_dist)

def most_common(lst):
    return max(set(lst), key=lst.count)

def visualize_graph(graph: nx.Graph, labels_dict: dict, title: str):
    fig = figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle(title, fontsize=24)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=150, node_color=list(labels_dict.values()))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, labels_dict, font_size=16, font_color='red')
    plt.show()

# returns graph modularity assuming that each node is in a separate community
def calculate_modularity(graph: nx.Graph):
    return nx.algorithms.community.modularity(graph, [{i} for i in range(len(graph.nodes))])
