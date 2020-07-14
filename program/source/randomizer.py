import random


class Randomizer:
    def __init__(self, data : list):
        self.data = data

    def choose(self):
        return random.choice(self.data)

    def sample(self, n = 1):
        return random.sample(self.data, n)

    @staticmethod
    def rand_int(min, max):
        return random.randint(min, max)