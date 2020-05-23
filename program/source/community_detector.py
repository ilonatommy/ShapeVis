from source import graph
from source.graph import Graph


class Community:
    def __init__(self):
        self.weights_total = 0
        self.weights_inside = 0
        self.nodes = []


class CommunityDetector:
    def __init__(self, graph: graph.Graph, landmarks, rev_neigh, weights):
        self.graph = graph
        self.landmarks = landmarks
        self.rev_neigh = rev_neigh
        # weights is always a symmetric matrix
        self.weights = weights
        self.communities = []
        # in initial partition there are as many communities as there are nodes:
        for node in self.graph.nodes:
            new_community = Community()
            new_community.nodes.append(node)
            self.communities.append(new_community)

    def __get_weight_between(self, node1, node2):
        try:
            w_n1_idx = self.landmarks[str(node1)]
        except:
            w_n1_idx = self.landmarks[self.rev_neigh[str(node1)]]
        try:
            w_n2_idx = self.landmarks[str(node2)]
        except:
            w_n2_idx = self.landmarks[self.rev_neigh[str(node2)]]
        return self.weights[w_n1_idx][w_n2_idx]

    def __set_community_weights(self, community):
        # the amount we will get is: tmp_weight = 2*w_inside_comm + 1*incident_weights
        tmp_weight = 0
        for node in community.nodes:
            neighbours = self.graph.adjacency_dict[str(node)]
            for neigh in neighbours:
                tmp_weight += self.__get_weight_between(node, neigh)
        community.weights_inside = self.__get_weight_inside_community(community)
        community.weights_total = tmp_weight - community.weights_inside

    def __get_weight_inside_community(self, community):
        total_sum = 0
        # if we have only 1 node in community, so no edges then return 0
        if len(community.nodes) < 2:
            return total_sum
        for node1 in community.nodes:
            for node2 in community.nodes:
                if not Graph.are_equal_nodes(node1, node2):
                    weight = self.__get_weight_between(node1, node2)
                    total_sum += weight
        # for 2 nodes we have 1 weight, which is added twice, but we need it only once
        return total_sum / 2



