import unittest
from unittest.mock import patch, Mock
import numpy as np

from source.witness_complex import WitnessComplexGraphBuilder
from source.randomizer import Randomizer
from source.data_processor import DataProcessor
from source.algo_comparer import AlgoComparer

TEST_DATA = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0,0], [3,2], [1.25, 0]])
TEST_LABELS = np.array([0, 1, 1, 0, 1, 0])
TEST_CLASSES = range(2)

TEST_SAMPLES_IDXS = [1, 0, 5]
TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ])]

class TestWitnessComplex(unittest.TestCase):

    @patch('source.randomizer.Randomizer')
    def setUp(self, MockRandomizer):
        mock_randomizer = MockRandomizer.return_value
        expected_nodes = TEST_SAMPLES
        mock_randomizer.sample.return_value = TEST_SAMPLES_IDXS

        stub_data_processor = DataProcessor()
        stub_data_processor.data = TEST_DATA
        stub_data_processor.labels = TEST_LABELS
        stub_data_processor.names = TEST_CLASSES
        
        self.sut = WitnessComplexGraphBuilder(stub_data_processor, len(TEST_SAMPLES))

    def test_knn_graph_creation(self):
        self.sut.build_knn()
        expected_edges = [('[1.75 1.75]', '[1.  1.5]'), ('[1.  1.5]', '[1.25 0.  ]')]
        edges = self.sut.get_graph().edges
        np.testing.assert_array_equal(edges, expected_edges)

    def test_knn_augmentation(self):
        self.sut.build_knn()
        self.sut.build_augmented_knn()

        expected_edges = [('[1.75 1.75]', '[1.  1.5]'), ('[1.75 1.75]', '[1.25 0.  ]'), ('[1.  1.5]', '[1.25 0.  ]')]
        edges = self.sut.get_graph().edges
        np.testing.assert_array_equal(edges, expected_edges)

        


if __name__ == '__main__':
    unittest.main()