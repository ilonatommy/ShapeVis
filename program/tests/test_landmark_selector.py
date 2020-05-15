import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.uniform_sampler import UniformSampler
from source.graph import Graph
from source.landmark_selector import LandmarkSelector

TEST_DATA = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0,0], [3,2], [1.25, 0], [0., 2.], [4., 0.], [2., 1], [3.5, 1], [0.5, 2.5], [0.5, 1]])
TEST_LABELS = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
TEST_CLASSES = range(2)
TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ]), np.array([4., 0.]), np.array([0.5, 2.5]), np.array([0.5, 1])]

def assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict):
    for key in expected_adjacency_dict.keys():
        np.testing.assert_array_equal(adjacency_dict[key], expected_adjacency_dict[key])

class TestLandmarkSelector(unittest.TestCase):
    
    def setUp(self):
        graph = Graph(TEST_SAMPLES)
        graph.adjacency_dict = {'[1.75 1.75]': [np.array([1. , 1.5]), np.array([1.25, 0.  ]), np.array([4., 0.])],
                                '[1.  1.5]': [np.array([0.5, 1. ]), np.array([1.75, 1.75]), np.array([0.5, 2.5])],
                                '[1.25 0.  ]': [np.array([0.5, 1. ]), np.array([1.75, 1.75])],
                                '[4. 0.]': [np.array([1.25, 0.  ]), np.array([1.75, 1.75])],
                                '[0.5 2.5]': [np.array([1. , 1.5])],
                                '[0.5 1. ]': [np.array([1. , 1.5]), np.array([1.25, 0.  ])]}
        self.sut = LandmarkSelector(graph)

    @patch('source.uniform_sampler.UniformSampler')
    def test_dummy(self, MockUniformSampler):
        LANDMARKS = [np.array([0.5, 2.5]),
                     np.array([0.5, 1. ]),
                     np.array([4., 0.])]
        mock_uniform_sampler = MockUniformSampler.return_value
        expected_nodes = TEST_SAMPLES
        mock_uniform_sampler.sample.side_effect = LANDMARKS

        self.sut.select_landmarks(l=1)
        landmarks = self.sut.get_landmarks()
        rev_neigh = self.sut.get_rev_neigh()
        expected_rev_neigh = {'[1.  1.5]': np.array([0.5, 2.5]),
                              '[1.25 0.  ]': np.array([0.5, 1. ]),
                              '[1.75 1.75]': np.array([4., 0.])}
        
        assert_adjacency_dicts_are_equal(rev_neigh, expected_rev_neigh)
        