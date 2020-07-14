from __future__ import division
from sklearn import datasets
import numpy as np

DIGITS_NO = 10

class DataProcessor:
    def __init__(self):
        self.data = []
        self.labels = []
        self.names = []

    def load_mnist(self, limit = 30):
        mnist = datasets.fetch_openml('mnist_784')
        self.data = mnist.data[:limit]
        self.labels = mnist.target[:limit]
        self.names = [i for i in range(DIGITS_NO)]
