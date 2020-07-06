import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.random_walker import RandomWalker
from source.randomizer import Randomizer
from source.graph import Graph
from source.landmark_selector import LandmarkSelector

TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1., 1.5]), np.array([1.25, 0.]), np.array([4., 0.]),
                 np.array([0.5, 2.5]), np.array([0.5, 1.]), np.array([3., 2.])]
TEST_LABELS = np.array([list(range(len(TEST_SAMPLES)))])
LANDMARKS = {'[0.5 2.5]': 0, '[0.5 1. ]': 1, '[4. 0.]': 2}

BETA  = 2
THETA = 2
ENDPOINTS = [np.array([0.5, 1.]), np.array([1.75, 1.75]),
             np.array([1.75, 1.75]), np.array([0.5, 2.5]),
             np.array([0.5, 1.]), np.array([1.75, 1.75])]


class TestLandmarkSelector(unittest.TestCase):

    def setUp(self):
        graph = Graph(TEST_SAMPLES, TEST_LABELS)
        graph.adjacency_dict = {
            '[1.75 1.75]': [np.array([1., 1.5]),    np.array([1.25, 0.]),   np.array([4., 0.]),  np.array([3., 2.])],
            '[1.  1.5]':   [np.array([0.5, 1.]),    np.array([1.75, 1.75]), np.array([0.5, 2.5])],
            '[1.25 0.  ]': [np.array([0.5, 1.]),    np.array([1.75, 1.75])],
            '[4. 0.]':     [np.array([1.25, 0.]),   np.array([1.75, 1.75]), np.array([3., 2.])],
            '[0.5 2.5]':   [np.array([1., 1.5]),    np.array([3., 2.])],
            '[0.5 1. ]':   [np.array([1., 1.5]),    np.array([1.25, 0.])],
            '[3. 2.]':     [np.array([1.75, 1.75]), np.array([0.5, 2.5]),   np.array([4., 0.])]}
        landmarks_cnt = len(LANDMARKS)
        beta = BETA
        theta1 = 1
        theta2 = 5
        self.sut = RandomWalker(graph, landmarks_cnt, beta, theta1, theta2)

    @patch('source.randomizer.Randomizer')
    def test_random_walk(self, MockRandomizer):
        MockRandomizer.rand_int.return_value = THETA
        mock_randomizer = MockRandomizer.return_value
        mock_randomizer.choose.side_effect = BETA*ENDPOINTS

        revNeigh = {'[1.  1.5]':   np.array([0.5, 2.5]),
                    '[3. 2.]':     np.array([0.5, 2.5]),
                    '[1.25 0.  ]': np.array([0.5, 1.]),
                    '[1.75 1.75]': np.array([4., 0.])}
        self.sut.walk(LANDMARKS, revNeigh)

        expected_n_matrix = np.array([np.array([0., 0., 2.]),
                                      np.array([2., 0., 0.]),
                                      np.array([0., 0., 2.])])
        n_matrix = self.sut.get_n_matrix()
        np.testing.assert_array_equal(n_matrix, expected_n_matrix)
        
        self.sut.calculate_weigths(2)
        expected_w_matrix = np.array([np.array([0., 1., 1.]),
                                      np.array([1., 0., 0.]),
                                      np.array([1., 0., 1.])])
        w_matrix = self.sut.get_w_matrix()
        np.testing.assert_array_equal(w_matrix, expected_w_matrix)
        
        self.sut.calculate_weigths(3)
        expected_w_matrix = np.array([np.array([0., 0., 0.]),
                                      np.array([0., 0., 0.]),
                                      np.array([0., 0., 0.])])
        w_matrix = self.sut.get_w_matrix()
        np.testing.assert_array_equal(w_matrix, expected_w_matrix)


if __name__ == '__main__':
    unittest.main()