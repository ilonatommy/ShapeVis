import mnist
import numpy as np


class DataProcessor:
    def __init__(self):
        pass

    def load_mnist(self):
        self.data = mnist.train_images().astype(np.float32) / 255.0
        self.labels = mnist.train_labels()
