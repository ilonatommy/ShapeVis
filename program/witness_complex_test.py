import unittest
from unittest.mock import patch, Mock
import numpy as np

from witness_complex import WitnessComplexCreator, UniformSampler
from data_processor import DataProcessor


class TestMainfoldLandmarker(unittest.TestCase):

    def setUp(self):
        stub_data_processor = DataProcessor()
        stub_data_processor.data = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0,0], [3,2], [1.25, 0]])
        stub_data_processor.labels = [0, 1, 1, 0, 1, 0]
        stub_data_processor.names = range(2)

        self.sut = WitnessComplexCreator(stub_data_processor, 3)

    @patch('mainfold_landmarker.UniformSampler')
    def test_dummy(self, MockUniformSampler):
        mock_uniform_sampler = MockUniformSampler()
        mock_uniform_sampler.sample.return_value = [np.array([1.75, 1.75]), np.array([1. , 1.5]), np.array([1.25, 0.  ])] 
        self.sut.create_knn_graph()
        
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()