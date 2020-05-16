import random


class RandomChoice:
    def __init__(self, data : list):
        self.data = data

    def choose(self, n = 1):
        return random.choice(self.data)
