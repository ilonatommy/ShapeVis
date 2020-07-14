import unittest
import networkx as nx
import community as community_louvain

from source.manifold_tearer import ManifoldTearer

GRAPH_SIZE = 10
NODES = list(range(GRAPH_SIZE))
EDGES = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (7, 8), (7, 9), (6, 3), (4, 6), (5, 6)]

class TestManifoldTearer(unittest.TestCase):
    def test_manifold_tearer(self):
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        labels = list(range(GRAPH_SIZE))

        ManifoldTearer.reduce_edges(graph, labels, -0.5)

if __name__ == '__main__':
    unittest.main()
