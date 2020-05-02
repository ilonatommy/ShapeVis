from __future__ import division
from sklearn import datasets
import numpy as np


class DataProcessor:
    def __init__(self):
        self.data = []
        self.labels = []
        self.names = []

    def load_mnist(self):
        LIMIT = 2000
        mnist = datasets.fetch_openml('mnist_784')
        self.data = mnist.data[:LIMIT]
        print(self.data.shape)
        print(self.data[0])
        self.labels = list(map(int, mnist.target[:LIMIT].tolist()))
        self.names = [i for i in range(10)]

    def load_artificial_data(self):
        self.data = np.array([[1, 1.5], [1.75, 1.75], [2.5, 0.5], [0.1,0], [3,2]])
        self.labels = [0, 0, 1, 1, 1]
        self.names = range(2)
