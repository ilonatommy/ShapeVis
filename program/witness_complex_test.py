import unittest
from unittest.mock import patch, Mock
import numpy as np

from witness_complex import WitnessComplexCreator
from uniform_sampler import UniformSampler
from data_processor import DataProcessor
from algo_comparer import AlgoComparer


TEST_DATA = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0,0], [3,2], [1.25, 0]])
TEST_LABELS = [0, 1, 1, 0, 1, 0]
TEST_CLASSES = range(2)
TEST_SAMPLES = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ])]

class TestWitnessComplex(unittest.TestCase):

    def setUp(self):
        stub_data_processor = DataProcessor()
        stub_data_processor.data = TEST_DATA
        stub_data_processor.labels = TEST_LABELS
        stub_data_processor.names = TEST_CLASSES
        
        self.sut = WitnessComplexCreator(stub_data_processor, len(TEST_SAMPLES))

    @patch('uniform_sampler.UniformSampler')
    def test_dummy(self, MockUniformSampler):
        mock_uniform_sampler = MockUniformSampler.return_value
        expected_nodes = TEST_SAMPLES
        mock_uniform_sampler.sample.return_value = expected_nodes
        nodes = self.sut.create_knn_graph()




if __name__ == '__main__':
    unittest.main()