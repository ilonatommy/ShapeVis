from __future__ import division
from sklearn import datasets


class DataProcessor:
    def __init__(self):
        pass

    def load_mnist(self):
        LIMIT = 2000
        mnist = datasets.fetch_openml('mnist_784')
        self.data = mnist.data[:LIMIT]
        self.labels = list(map(int, mnist.target[:LIMIT].tolist()))
        self.names = [i for i in range(10)]
