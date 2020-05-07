import random


class UniformSampler:
    def __init__(self, data : list):
        self.data = data

    def sample(self, n = 1):
        return random.sample(self.data, n)
