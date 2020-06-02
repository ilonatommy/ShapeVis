from source import graph, randomizer


class LandmarkSelector:

    def __init__(self, graph: graph.Graph):
        self.graph = graph
        self.landmarks = dict()
        self.rev_neigh = dict()

    def select_landmarks(self, l: int):
        l_idx = 0
        while len(self.graph.nodes) > 0:
            landmark = randomizer.Randomizer(list(self.graph.nodes)).sample()
            self.landmarks[str(landmark)] = l_idx

            # TODO uzupełnić słownik, aby byli widoczni sąsiedzi z drugiej strony
            for neigh in self.graph.adjacency_dict[str(landmark)]:
                self.rev_neigh[str(neigh)] = landmark
                self.graph.remove_node(neigh)

            self.graph.remove_node(landmark)
            l_idx = l_idx + 1

    def get_landmarks(self):
        return self.landmarks

    def get_rev_neigh(self):
        return self.rev_neigh
