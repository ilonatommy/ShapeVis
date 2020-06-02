import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.randomizer import Randomizer
from source.graph import Graph
from source.landmark_selector import LandmarkSelector

TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ]), np.array([4., 0.]), np.array([0.5, 2.5]), np.array([0.5, 1]), np.array([3, 2])]

def assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict):
    for key in expected_adjacency_dict.keys():
        np.testing.assert_array_equal(adjacency_dict[key], expected_adjacency_dict[key])

class TestLandmarkSelector(unittest.TestCase):
    
    def setUp(self):
        graph = Graph(TEST_SAMPLES)
        graph.adjacency_dict = {'[1.75 1.75]': [np.array([1. , 1.5]), np.array([1.25, 0.  ]), np.array([4., 0.]), np.array([3, 2])],
                                '[1.  1.5]': [np.array([0.5, 1. ]), np.array([1.75, 1.75]), np.array([0.5, 2.5])],
                                '[1.25 0.  ]': [np.array([0.5, 1. ]), np.array([1.75, 1.75])],
                                '[4. 0.]': [np.array([1.25, 0.  ]), np.array([1.75, 1.75]), np.array([3, 2])],
                                '[0.5 2.5]': [np.array([1. , 1.5]), np.array([3, 2])],
                                '[0.5 1. ]': [np.array([1. , 1.5]), np.array([1.25, 0.  ])],
                                '[3 2]': [np.array([1.75 , 1.75]), np.array([0.5, 2.5]), np.array([4, 0])]}
        self.sut = LandmarkSelector(graph)

    @patch('source.randomizer.Randomizer')
    def test_landmark_selection(self, MockRandomizer):
        LANDMARKS = [np.array([0.5, 2.5]), np.array([0.5, 1. ]), np.array([4., 0.])]
        #LANDMARKS = {'[0.5 2.5]': 0, '[0.5 1. ]': 1, '[4. 0.]': 2} #prawdziwe landmarki, ale nie do mockowania samplera
        mock_randomizer = MockRandomizer.return_value
        expected_nodes = TEST_SAMPLES
        mock_randomizer.sample.side_effect = LANDMARKS

        self.sut.select_landmarks(l=1)
        landmarks = self.sut.get_landmarks()
        rev_neigh = self.sut.get_rev_neigh()
        expected_rev_neigh = {'[1.  1.5]': np.array([0.5, 2.5]),
                              '[1.25 0.  ]': np.array([0.5, 1. ]),
                              '[1.75 1.75]': np.array([4., 0.]),
                              '[3 2]': np.array([0.5, 2.5])}
        
        assert_adjacency_dicts_are_equal(rev_neigh, expected_rev_neigh)


if __name__ == '__main__':
    unittest.main()