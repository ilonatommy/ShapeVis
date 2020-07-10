from __future__ import division
from sklearn import datasets
import numpy as np


class DataProcessor:
    def __init__(self):
        self.data = []
        self.labels = []
        self.names = []

    def load_mnist(self):
        LIMIT = 30
        mnist = datasets.fetch_openml('mnist_784')
        self.data = mnist.data[:LIMIT]
        self.labels = mnist.target[:LIMIT]
        self.names = [i for i in range(10)]
