import numpy as np
import random
from source import graph, randomizer


class RandomWalker:

    def __init__(self, graph: graph.Graph, landmarks_cnt, beta, theta1, theta2):
        self.beta = beta
        self.theta1 = theta1
        self.theta2 = theta2
        self.graph = graph
        # to save num of random walks started at Li and finishing at Lj:
        self.n_matrix = np.zeros((landmarks_cnt, landmarks_cnt))
        self.a_matrix = np.zeros((landmarks_cnt, landmarks_cnt))
        self.w_matrix = np.zeros((landmarks_cnt, landmarks_cnt))

    def __walk_landmark(self, landmark, theta):
        current_node = landmark
        for step in range(theta):
            rand_choice = randomizer.Randomizer(self.graph.adjacency_dict[str(current_node)])
            current_node = rand_choice.choose()
        return current_node

    def walk(self, landmarks, rev_neigh):
        for time in range(self.beta):
            for landmark in landmarks:
                theta = randomizer.Randomizer.rand_int(self.theta1, self.theta2)
                endpoint = self.__walk_landmark(landmark, theta)
                try: # nie każdy endpoint jest tutaj w revNeigh - niektóre są landmarkami
                    l_j = rev_neigh[str(endpoint)]
                    j = landmarks[str(l_j)]
                except:
                    j = landmarks[str(endpoint)]
                i = landmarks[landmark]
                self.n_matrix[i][j] = self.n_matrix[i][j] + 1

    def calculate_weigths(self, threshold):
        self.a_matrix = np.where(self.n_matrix < threshold, 0, self.n_matrix) / self.n_matrix.sum(axis=1)
        self.w_matrix = self.a_matrix + self.a_matrix.T - self.a_matrix * self.a_matrix.T

    def get_n_matrix(self):
        return self.n_matrix

    def get_a_matrix(self):
        return self.a_matrix

    def get_w_matrix(self):
        return self.w_matrix