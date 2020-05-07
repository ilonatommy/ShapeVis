import graph
import uniform_sampler


class LandmarkSelector:

    def __init__(self, graph : graph.Graph):
        self.graph = graph
        self.landmarks = []
        self.rev_neigh = dict()

    def select_landmarks(self, l : int):
        while len(self.graph.nodes) > 0:
            landmark = uniform_sampler.UniformSampler(list(self.graph.nodes)).sample()
            self.landmarks.append(landmark)

            # TODO uzupełnić słownik, aby byli widoczni sąsiedzi z drugiej strony
            for neigh in self.graph.adjacency_dict[str(landmark)]:
                self.rev_neigh[str(neigh)] = landmark
                self.graph.remove_node(neigh)
            
            self.graph.remove_node(landmark)
    def get_landmarks(self):
        return self.landmarks

    def get_rev_neigh(self):
        return self.rev_neigh