from source import randomizer, helpers
import networkx as nx


class LandmarkSelector:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.landmarks = dict()
        self.rev_neigh = dict()

    def select_landmarks(self, l: int):
        l_idx = 0
        while len(self.graph.nodes) > 0:
            landmark = randomizer.Randomizer(list(self.graph.nodes)).sample()
            self.landmarks[str(landmark)] = l_idx

            landmark_neighbors_info = self.graph[str(landmark)].items()
            landmark_neighbors = list(map(lambda x: x[0], list(landmark_neighbors_info)))
            for neigh in landmark_neighbors:
                self.rev_neigh[neigh] = landmark

            self.graph.remove_nodes_from(landmark_neighbors)
            self.graph.remove_node(str(landmark))
            l_idx = l_idx + 1

    def get_landmarks(self):
        return self.landmarks

    def get_rev_neigh(self):
        return self.rev_neigh
