import unittest
from unittest.mock import patch, Mock
import numpy as np
import networkx as nx

from source.randomizer import Randomizer
from source.landmark_selector import LandmarkSelector

TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ]), np.array([4., 0.]), np.array([0.5, 2.5]), np.array([0.5, 1]), np.array([3, 2])]
TEST_LABELS = np.array(list(range(len(TEST_SAMPLES))))

INPUT_EDGES = [('[1.75 1.75]', '[1.  1.5]'),
               ('[1.75 1.75]', '[1.25 0.  ]'),
               ('[1.75 1.75]', '[4. 0.]'),
               ('[1.75 1.75]', '[3 2]'),
               ('[1.  1.5]', '[0.5 1. ]'),
               ('[1.  1.5]', '[0.5 2.5]'),
               ('[1.25 0.  ]', '[0.5 1. ]'),
               ('[4. 0.]', '[3 2]'),
               ('[0.5 2.5]', '[3 2]')]

def assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict):
    for key in expected_adjacency_dict.keys():
        np.testing.assert_array_equal(adjacency_dict[key], expected_adjacency_dict[key])



class TestLandmarkSelector(unittest.TestCase):
    
    def setUp(self):
        graph = nx.Graph()
        nds = map(lambda i: (str(TEST_SAMPLES[i]), {"indices":TEST_SAMPLES[i], "label": TEST_LABELS[i]}), range(len(TEST_SAMPLES)))
        graph.add_nodes_from(list(nds))
        graph.add_edges_from(INPUT_EDGES)
        self.sut = LandmarkSelector(graph)

    @patch('source.randomizer.Randomizer')
    def test_landmark_selection(self, MockRandomizer):
        LANDMARKS = [np.array([0.5, 2.5]), np.array([0.5, 1. ]), np.array([4., 0.])]
        #LANDMARKS = {'[0.5 2.5]': 0, '[0.5 1. ]': 1, '[4. 0.]': 2} #prawdziwe landmarki, ale nie do mockowania samplera
        mock_randomizer = MockRandomizer.return_value
        mock_randomizer.choose.side_effect = LANDMARKS

        self.sut.select_landmarks()
        landmarks = self.sut.get_landmarks()
        rev_neigh = self.sut.get_rev_neigh()
        expected_rev_neigh = {'[1.  1.5]': np.array([0.5, 2.5]),
                              '[1.25 0.  ]': np.array([0.5, 1. ]),
                              '[1.75 1.75]': np.array([4., 0.]),
                              '[3 2]': np.array([0.5, 2.5])}
        
        assert_adjacency_dicts_are_equal(rev_neigh, expected_rev_neigh)


if __name__ == '__main__':
    unittest.main()