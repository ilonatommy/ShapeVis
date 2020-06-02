import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.witness_complex import WitnessComplexGraphBuilder
from source.randomizer import Randomizer
from source.data_processor import DataProcessor
from source.algo_comparer import AlgoComparer

TEST_DATA = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0,0], [3,2], [1.25, 0]])
TEST_LABELS = [0, 1, 1, 0, 1, 0]
TEST_CLASSES = range(2)
TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ])]

def assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict):
    for key in expected_adjacency_dict.keys():
        np.testing.assert_array_equal(adjacency_dict[key], expected_adjacency_dict[key])

class TestWitnessComplex(unittest.TestCase):

    @patch('source.randomizer.Randomizer')
    def setUp(self, MockRandomizer):
        mock_randomizer = MockRandomizer.return_value
        expected_nodes = TEST_SAMPLES
        mock_randomizer.sample.return_value = expected_nodes

        stub_data_processor = DataProcessor()
        stub_data_processor.data = TEST_DATA
        stub_data_processor.labels = TEST_LABELS
        stub_data_processor.names = TEST_CLASSES
        
        self.sut = WitnessComplexGraphBuilder(stub_data_processor, len(TEST_SAMPLES))

    def test_knn_graph_creation(self):
        self.sut.build_knn()
        expected_adjacency_dict = {'[1.75 1.75]': [np.array([1. , 1.5])],
                                   '[1.  1.5]': [np.array([1.75, 1.75])],
                                   '[1.25 0.  ]': [np.array([1. , 1.5])]}
        adjacency_dict = self.sut.get_graph().adjacency_dict
        assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict)

    def test_knn_augmentation(self, ):
        self.sut.build_knn()
        self.sut.build_augmented_knn()
        adjacency_dict = self.sut.get_graph().adjacency_dict
        expected_adjacency_dict = {'[1.75 1.75]': [np.array([1. , 1.5]), np.array([1.25, 0.  ])],
                                   '[1.  1.5]': [np.array([1.75, 1.75]), np.array([1.25, 0.  ])],
                                   '[1.25 0.  ]': [np.array([1. , 1.5]), np.array([1.75, 1.75])]}
        assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict)

        


if __name__ == '__main__':
    unittest.main()