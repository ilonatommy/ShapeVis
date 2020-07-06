from graph import Graph


class NodesCoder:
    def __init__(self):
        self.coder = {}
        self.decoder = {}

    def rename_nodes_to_numbers(self, graph: Graph, landmarks, rev_neigh):
        coded_numbers = []
        coded_number = 0
        for node in graph.nodes:
            self.coder[str(node)] = coded_number
            self.decoder[coded_number] = node
            coded_numbers.append(coded_number)
            coded_number += 1
        coded_adjacency_dictionary = {}
        for node in graph.adjacency_dict:
            neighbours = graph.adjacency_dict[str(node)]
            coded_neighbours = []
            for n in neighbours:
                coded_neighbours.append(self.coder[str(n)])
            coded_adjacency_dictionary[self.coder[str(node)]] = coded_neighbours
        coded_landmarks = {}
        for landmark in landmarks:
            coded_landmarks[self.coder[str(landmark)]] = landmarks[str(landmark)]
        coded_rev_neigh = {}
        for rn in rev_neigh:
            coded_rev_neigh[self.coder[str(rn)]] = self.coder[str(rev_neigh[str(rn)])]
        coded_graph = Graph(coded_numbers)
        coded_graph.adjacency_dict = coded_adjacency_dictionary
        return coded_graph, coded_landmarks, coded_rev_neigh
