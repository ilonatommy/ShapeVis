import random


class UniformSampler:
    def __init__(self, data : list):
        self.data = data

    def sample(self, n):
        return random.sample(self.data, n)
