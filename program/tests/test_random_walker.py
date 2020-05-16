import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.random_walker import RandomWalker
from source.uniform_sampler import UniformSampler
from source.graph import Graph
from source.landmark_selector import LandmarkSelector

TEST_DATA = np.array(
    [[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0, 0], [3, 2], [1.25, 0], [0., 2.], [4., 0.], [2., 1], [3.5, 1], [0.5, 2.5],
     [0.5, 1]])
TEST_LABELS = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
TEST_CLASSES = range(2)
TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]),
                 np.array([0.5, 2.5]), np.array([0.5, 1.]), np.array([3., 2.])]
LANDMARKS = {'[0.5 2.5]': 0, '[0.5 1. ]': 1, '[4. 0.]': 2}


def assert_matrices_are_equal(matrix1, matrix2):
    np.testing.assert_array_equal(matrix1, matrix2)


class TestLandmarkSelector(unittest.TestCase):

    def setUp(self):
        graph = Graph(TEST_SAMPLES)
        graph.adjacency_dict = {
            '[1.75 1.75]': [np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]), np.array([3., 2.])],
            '[1.  1.5]': [np.array([0.5, 1.]), np.array([1.75, 1.75]), np.array([0.5, 2.5])],
            '[1.25 0.  ]': [np.array([0.5, 1.]), np.array([1.75, 1.75])],
            '[4. 0.]': [np.array([1.25, 0.]), np.array([1.75, 1.75]), np.array([3., 2.])],
            '[0.5 2.5]': [np.array([1., 1.5]), np.array([3., 2.])],
            '[0.5 1. ]': [np.array([1., 1.5]), np.array([1.25, 0.])],
            '[3. 2.]': [np.array([1.75, 1.75]), np.array([0.5, 2.5]), np.array([4., 0.])]}
        landmarks_cnt = len(LANDMARKS)
        beta = 3
        theta1 = 3
        theta2 = 6
        self.sut = RandomWalker(graph, landmarks_cnt, beta, theta1, theta2)

    @patch('source.random_choice.RandomChoice')
    def test_dummy(self, MockRandomChoice):
        revNeigh = {'[1.  1.5]': np.array([0.5, 2.5]),
                    '[3. 2.]': np.array([0.5, 2.5]),
                    '[1.25 0.  ]': np.array([0.5, 1.]),
                    '[1.75 1.75]': np.array([4., 0.])}

        #jak zamockować poprawnie RandomChoice?
        mock_random_choice = MockRandomChoice.return_value
        expected_nodes = TEST_SAMPLES
        mock_random_choice.choose.side_effect = [np.array([1., 1.5]), np.array([1., 1.5]), np.array([1.25, 0.])]

        self.sut.walk(LANDMARKS, revNeigh)
        self.sut.calculate_weigths(2)
        w_matrix = self.sut.get_w_matrix()
        print("W:\n", w_matrix)
        expected_rev_neigh = {'[1.  1.5]': np.array([0.5, 2.5]),
                              '[1.25 0.  ]': np.array([0.5, 1.]),
                              '[1.75 1.75]': np.array([4., 0.]),
                              '[3 2]': np.array([0.5, 2.5])}
        assert_matrices_are_equal(w_matrix, w_matrix)


if __name__ == '__main__':
    unittest.main()