import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.graph import Graph

TEST_NODES = [np.array([0., 0.]),
              np.array([0., 1.]),
              np.array([-1., -0.5])]

OTHER_NODE = np.array([0., 0., 0.])

class TestGraph(unittest.TestCase):

    def setUp(self):
        self.sut = Graph(TEST_NODES)

    def test_calculate_distance(self):
        self.assertEqual(self.sut.calculate_distance(self.sut.nodes[0], self.sut.nodes[1]), np.sqrt(1))
        self.assertEqual(self.sut.calculate_distance(self.sut.nodes[0], self.sut.nodes[2]), np.sqrt(1.25))
        self.assertEqual(self.sut.calculate_distance(self.sut.nodes[1], self.sut.nodes[2]), np.sqrt(3.25))

    def test_calculate_distance_different_shapes(self):
        self.assertEqual(self.sut.calculate_distance(self.sut.nodes[1], OTHER_NODE), -1)

    def test_checking_if_nodes_are_equal(self):
        self.assertEqual(self.sut.are_equal_nodes(TEST_NODES[0], TEST_NODES[0]), True)
        self.assertEqual(self.sut.are_equal_nodes(TEST_NODES[0], TEST_NODES[1]), False)
        self.assertEqual(self.sut.are_equal_nodes(TEST_NODES[0], OTHER_NODE), False)

    def test_checking_if_nodes_are_neighbours(self):
        self.sut.adjacency_dict = {'[0. 0.]': [np.array([0., 1.])],
                                   '[0. 1.]': [np.array([0., 0.])],
                                   '[-1.  -0.5]': [np.array([0., 0.])]}
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[0], TEST_NODES[1]), True)
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[0], TEST_NODES[2]), False)
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[1], TEST_NODES[0]), True)
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[1], TEST_NODES[2]), False)
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[2], TEST_NODES[0]), True)
        self.assertEqual(self.sut.is_node2_neighbor_of_node1(TEST_NODES[2], TEST_NODES[1]), False)

    def test_checking_number_of_neighbors(self):
        self.sut.adjacency_dict = {'[0. 0.]': [np.array([0., 1.]), np.array([-1., -0.5])],
                                   '[0. 1.]': [np.array([0., 0.])],
                                   '[-1.  -0.5]': [np.array([0., 0.])]}
        self.assertEqual(self.sut.has_node_less_than_k_neighbors(TEST_NODES[0], 2), False)
        self.assertEqual(self.sut.has_node_less_than_k_neighbors(TEST_NODES[1], 2), True)
        self.assertEqual(self.sut.has_node_less_than_k_neighbors(TEST_NODES[2], 0), False)
        
    def test_checking_if_node_is_in_graph(self):
        self.assertEqual(self.sut.has_node(TEST_NODES[0]), True)
        self.assertEqual(self.sut.has_node(OTHER_NODE), False)

    
if __name__ == '__main__':
    unittest.main()