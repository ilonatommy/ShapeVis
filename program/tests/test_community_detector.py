import unittest
import numpy as np

from community_detector import CommunityDetector
from graph import Graph
from nodes_coder import NodesCoder

TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]),
                np.array([0.5, 2.5]), np.array([0.5, 1.]), np.array([3., 2.])]
LANDMARKS = {'[0.5 2.5]': 0, '[0.5 1. ]': 1, '[4. 0.]': 2}
REV_NEIGH = {'[1.  1.5]': np.array([0.5, 2.5]),
             '[3. 2.]': np.array([0.5, 2.5]),
             '[1.25 0.  ]': np.array([0.5, 1.]),
             '[1.75 1.75]': np.array([4., 0.])}
LANDMARK_WEIGHTS = [[0.8, 0.5, 0.6],
                    [0.5, 0, 0.1],
                    [0.6, 0.1, 0.2]]


def assert_matrices_are_equal(matrix1, matrix2):
    np.testing.assert_array_equal(matrix1, matrix2)


class TestCommunityDetectors(unittest.TestCase):

    def setUp(self):
        graph = Graph(TEST_SAMPLES)
        graph.adjacency_dict = {
            '[1.75 1.75]': [np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]), np.array([3., 2.])],
            '[1.  1.5]': [np.array([0.5, 1.]), np.array([1.75, 1.75]), np.array([0.5, 2.5])],
            '[1.25 0.  ]': [np.array([0.5, 1.]), np.array([1.75, 1.75]), np.array([4., 0.]),],
            '[4. 0.]': [np.array([1.25, 0.]), np.array([1.75, 1.75]), np.array([3., 2.])],
            '[0.5 2.5]': [np.array([1., 1.5]), np.array([3., 2.])],
            '[0.5 1. ]': [np.array([1., 1.5]), np.array([1.25, 0.])],
            '[3. 2.]': [np.array([1.75, 1.75]), np.array([0.5, 2.5]), np.array([4., 0.])]}
        # print(graph)
        coder = NodesCoder() # codes string nodes into integer numbers 0...(len(nodes))
        graph, landmarks, rev_neigh = coder.rename_nodes_to_numbers(graph, LANDMARKS, REV_NEIGH)
        self.sut = CommunityDetector(graph, landmarks, rev_neigh, LANDMARK_WEIGHTS)

    def test_dummy(self):
        graph, weights = self.sut.run()
        EXPECTED_WEIGHTS = [[]]
        print(weights)
        # assert_matrices_are_equal(weights, EXPECTED_WEIGHTS)

    def test_debug(self):
        pass
        # for community in self.sut.communities:
        #     print(community.nodes)
        #     print("\t", community.get_community_weights(True))

if __name__ == '__main__':
    unittest.main()
