import unittest
from unittest.mock import patch
import numpy as np

from source.community_detector import CommunityDetector
from source.graph import Graph

TEST_DATA = np.array(
    [[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0, 0], [3, 2], [1.25, 0], [0., 2.], [4., 0.], [2., 1], [3.5, 1], [0.5, 2.5],
     [0.5, 1]])
TEST_LABELS = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
TEST_CLASSES = range(2)
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
        graph = Graph(TEST_SAMPLES, TEST_LABELS)
        graph.adjacency_dict = {
            '[1.75 1.75]': [np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]), np.array([3., 2.])],
            '[1.  1.5]': [np.array([0.5, 1.]), np.array([1.75, 1.75]), np.array([0.5, 2.5])],
            '[1.25 0.  ]': [np.array([0.5, 1.]), np.array([1.75, 1.75]), np.array([4., 0.]),],
            '[4. 0.]': [np.array([1.25, 0.]), np.array([1.75, 1.75]), np.array([3., 2.])],
            '[0.5 2.5]': [np.array([1., 1.5]), np.array([3., 2.])],
            '[0.5 1. ]': [np.array([1., 1.5]), np.array([1.25, 0.])],
            '[3. 2.]': [np.array([1.75, 1.75]), np.array([0.5, 2.5]), np.array([4., 0.])]}
        self.sut = CommunityDetector(graph, LANDMARKS, REV_NEIGH, LANDMARK_WEIGHTS)

    def test_dummy(self):
       graph, weights = self.sut.run()
       EXPECTED_WEIGHTS = [[0.,  0.6, 0.1, 0.2, 0.,  0. , 0.6],
                            [0.6, 0.,  0.,  0.,  0.8, 0.5,  0. ],
                            [0.1, 0.,  0.,  0.1, 0.,  0.,  0. ],
                            [0.2, 0.,  0.1, 0.,  0.,  0.,  0.6],
                            [0.,  0.8, 0.,  0.,  0.,  0.,  0.8],
                            [0.,  0.5,  0.,  0.,  0.,  0.,  0. ],
                            [0.6, 0.,  0.,  0.6, 0.8, 0.,  0. ]]
       assert_matrices_are_equal(weights, EXPECTED_WEIGHTS)

if __name__ == '__main__':
    unittest.main()
