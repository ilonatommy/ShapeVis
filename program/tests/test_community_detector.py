import unittest
import numpy as np
import networkx as nx

from source.community_detector import CommunityDetector

WEIGHT_MATRIX = np.array([np.array([1., 0.55,   0.65,   0.6,    0.55  ]),
                          np.array([0.55,   0.6975, 0.2,    0.2,    0.15  ]),
                          np.array([0.65,  0.2,    0.2775, 0.1,    0.15  ]),
                          np.array([0.6,    0.2,    0.1,   0.19,   0.15  ]),
                          np.array([0.55,  0.15,   0.15,   0.15,   0.    ])])
TEST_LABELS = [0, 1, 0, 1, 1]

LANDMARK_WEIGHTS = [[0.8, 0.5, 0.6],
                    [0.5, 0, 0.1],
                    [0.6, 0.1, 0.2]]

# TODO - now only for debuuging purposes, there should be some expectations added
class TestCommunityDetectors(unittest.TestCase):
    def test_community_detector(self):
        CommunityDetector.detect_communities(WEIGHT_MATRIX, labels, visualize=True)

if __name__ == '__main__':
    unittest.main()