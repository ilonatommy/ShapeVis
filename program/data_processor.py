from __future__ import division
from sklearn import datasets
import numpy as np


class DataProcessor:
    def __init__(self):
        self.data = []
        self.labels = []
        self.names = []

    def load_mnist(self):
        LIMIT = 20
        mnist = datasets.fetch_openml('mnist_784')
        self.data = mnist.data[:LIMIT]
        self.labels = list(map(int, mnist.target[:LIMIT].tolist()))
        self.names = [i for i in range(10)]
