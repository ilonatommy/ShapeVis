import unittest
from unittest.mock import patch, Mock
import numpy as np

from witness_complex import Graph

TEST_NODES = [np.array([0., 0.]),
              np.array([0., 1.]),
              np.array([-1., -0.5  ])]

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


if __name__ == '__main__':
    unittest.main()