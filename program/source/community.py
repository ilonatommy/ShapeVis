from source import graph
from source.graph import Graph


class Community:
    def __init__(self, graph: graph.Graph, landmarks, rev_neigh, weights):
        self.landmarks = landmarks
        self.rev_neigh = rev_neigh
        self.weights = weights
        self.graph = graph

        self.weights_total = 0
        self.weights_inside = 0
        self.nodes = []

    def get_landmarks_weight_between(self, node1, node2):
        try:
            w_n1_idx = self.landmarks[str(self.rev_neigh[str(node1)])]
        except:
            w_n1_idx = self.landmarks[str(node1)]
        try:
            w_n2_idx = self.landmarks[str(self.rev_neigh[str(node2)])]
        except:
            w_n2_idx = self.landmarks[str(node2)]
        return self.weights[w_n1_idx][w_n2_idx]

    def get_weights_between(self, node1, node2):
        return self.weights[node1][node2]

    def __get_weight_inside_community(self, from_landmarks):
        total_sum = 0
        # if we have only 1 node in community, so no edges then return 0
        if len(self.nodes) < 2:
            return total_sum
        for node1 in self.nodes:
            for node2 in self.nodes:
                # if the nodes are connected then check what is the weight of the connection:
                if self.graph.is_node2_neighbor_of_node1(node1, node2):
                    if from_landmarks:
                        weight = self.get_landmarks_weight_between(node1, node2)
                    else:
                        weight = self.get_weights_between(node1[0], node2[0])
                    total_sum += weight
        # for 2 nodes we have 1 weight, which is added twice, but we need it only once
        return total_sum / 2

    def get_community_weights(self, from_landmarks):
        # the amount we will get is: tmp_weight = 2*w_inside_comm + 1*incident_weights
        tmp_weight = 0
        for node in self.nodes:
            neighbours = self.graph.adjacency_dict[str(node)]
            for neigh in neighbours:
                if from_landmarks:
                    tmp_weight += self.get_landmarks_weight_between(node, neigh)
                else:
                    tmp_weight += self.get_weights_between(node[0], neigh[0])
        weights_inside = self.__get_weight_inside_community(from_landmarks)
        weights_total = tmp_weight - weights_inside
        return weights_inside, weights_total

    def get_weights_between_node_and_community(self, neighbours, node_i, from_landmarks): # returns k_i,in
        total_weight = 0
        # TODO check if the calculations are correct after changes
        coeff = 0.5
        intersection_list = []
        for node1 in self.nodes:
            for node2 in neighbours:
                if self.graph.are_equal_nodes(node1, node2):
                    intersection_list.append(node1)

        for community_neigh in intersection_list:
            if from_landmarks:
                total_weight += self.get_landmarks_weight_between(node_i, community_neigh)
            else:
                total_weight += self.get_weights_between(node_i[0], community_neigh[0])
        return total_weight * coeff

    def get_weights_of_incident_nodes(self, neighbours, node_i, from_landmarks):
        total_weight = 0
        for neigh in neighbours:
            if from_landmarks:
                total_weight += self.get_landmarks_weight_between(node_i, neigh)
            else:
                total_weight += self.get_weights_between(node_i[0], neigh[0])
        return total_weight

    def deep_copy(self):
        community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        community.nodes = list.copy(self.nodes)
        community.weights_total = self.weights_total
        community.weights_inside = self.weights_inside
        return community