import numpy as np

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
            w_n1_idx = self.landmarks[self.rev_neigh[str(node1)]]
        except:
            w_n1_idx = self.landmarks[str(node1)]
        try:
            w_n2_idx = self.landmarks[self.rev_neigh[str(node2)]]
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
                if not Graph.are_equal_nodes(node1, node2):
                    if from_landmarks:
                        weight = self.get_landmarks_weight_between(node1, node2)
                    else:
                        weight = self.get_weights_between(node1, node2)
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
                    tmp_weight += self.get_weights_between(node, neigh)
        weights_inside = self.__get_weight_inside_community(from_landmarks)
        weights_total = tmp_weight - weights_inside
        return weights_inside, weights_total

    def get_weights_between_node_and_community(self, neighbours, node_i): # returns k_i,in
        total_weight = 0
        intersection_list = list(set(self.nodes) & set(neighbours))
        for community_neigh in intersection_list:
            total_weight += self.get_landmarks_weight_between(node_i, community_neigh)
        return total_weight

    def get_weights_of_incident_nodes(self, neighbours, node_i):
        total_weight = 0
        for neigh in neighbours:
            total_weight += self.get_landmarks_weight_between(node_i, neigh)
        return total_weight

    def deep_copy(self):
        community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        community.nodes = self.nodes
        community.weights_total = self.weights_total
        community.weights_inside = self.weights_inside
        return community


class CommunityDetector:
    def __init__(self, graph: graph.Graph, landmarks, rev_neigh, weights, from_landmarks=True):
        self.graph = graph
        self.landmarks = landmarks
        self.rev_neigh = rev_neigh
        # weights is always a symmetric matrix
        self.weights = weights
        self.communities = []
        self.communities_dict = {}
        self.m = self.__get_accumulated_network_weights(from_landmarks)
        # in initial partition there are as many communities as there are nodes:
        community_idx = 0
        for node in self.graph.nodes:
            new_community = Community(graph, landmarks, rev_neigh, weights)
            new_community.nodes.append(node)
            self.communities.append(new_community)
            # dict stores info in which community to look for a node: complexity of finding community - O(2)
            self.communities_dict[(str(node))] = community_idx
            community_idx += 1

    def __get_accumulated_network_weights(self, from_landmarks):
        tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        total_weight = 0
        for node in self.graph.nodes:
            neighbour = self.graph.adjacency_dict[str(node)]
            for neigh in neighbour:
                if from_landmarks:
                    total_weight += tmp_community.get_landmarks_weight_between(neigh, node)
                else:
                    total_weight += tmp_community.get_weights_between(neigh, node)
        return total_weight / 2

    def __check_modularity_gain(self, node, dest_community: Community, from_landmarks):
        dest_community_cpy = dest_community.deep_copy()
        dest_community_cpy.nodes.append(node)
        w_inside, w_total = dest_community_cpy.get_community_weights(from_landmarks)
        node_neighbours = self.graph.adjacency_dict[str(node)]
        k_i_in = dest_community_cpy.get_weights_between_node_and_community(node_neighbours, node)
        k_i = dest_community_cpy.get_weights_of_incident_nodes(node_neighbours, node)
        d_Q = (k_i_in - 2*w_total * k_i) / (2 * self.m) # czy na pewno taki wzor?
        return d_Q

    @staticmethod
    def __shift_node_from_community(source_community: Community, dest_community: Community, node, from_landmarks):
        source_community.nodes.remove(node)
        source_community.weights_inside, source_community.weights_total = \
            source_community.get_community_weights(from_landmarks)
        dest_community.nodes.append(node)
        dest_community.weights_inside, dest_community.weights_total = \
            dest_community.get_community_weights(from_landmarks)

    def __fit_communities(self, from_landmarks):
        repeat = True
        iteration_idx = 0
        while(repeat):
            repeat = False
            for node in self.graph.nodes:
                src_community_idx = self.communities_dict[str(node)]
                src_community = self.communities[src_community_idx]
                neighbours = self.graph.adjacency_dict[str(node)]
                d_Q = []
                for neigh in neighbours:
                    community_idx = self.communities_dict[str(neigh)]
                    neighbour_community = self.communities[community_idx]
                    d_Q.append(self.__check_modularity_gain(node, neighbour_community, from_landmarks))
                max_dQ = max(d_Q)
                if max_dQ > 0:
                    repeat = True
                    iteration_idx += 1
                    neigh = neighbours[d_Q.index(max_dQ)]
                    community_idx = self.communities_dict[str(neigh)]
                    dest_community = self.communities[community_idx]
                    self.__shift_node_from_community(src_community, dest_community, node, from_landmarks)
        return iteration_idx

    def __get_graph_from_communities(self, from_landmarks):
        communities_number = len(self.communities)
        new_graph = Graph(range(communities_number))
        new_weights = np.zeros(communities_number, communities_number)
        tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)

        for n1 in new_graph.nodes:
            for n2 in new_graph.nodes:
                # if they are from different communities:
                c1_idx = self.communities_dict[str(n1)]
                c2_idx = self.communities_dict[str(n2)]
                if c1_idx != c2_idx:
                    # and are neighbours for each other:
                    if self.graph.adjacency_dict[str(n1)].any == n2:
                        # new nodes made from these communities will be neighbours:
                        new_graph.adjacency_dict[str(c1_idx)].append(c2_idx)
                        new_graph.adjacency_dict[str(c2_idx)].append(c1_idx)
                        # find weights between n1 and n2
                        if from_landmarks:
                            weight = tmp_community.get_landmarks_weight_between(n1, n2)
                        else:
                            weight = tmp_community.get_weights_between(n1, n2)
                        new_weights[c1_idx][c2_idx] += weight
                        new_weights[c2_idx][c1_idx] += weight

        return CommunityDetector(new_graph, self.landmarks, self.rev_neigh, new_weights, from_landmarks=False)

    def __deep_copy(self, src_detector):
        self.graph = src_detector.graph
        self.landmarks = src_detector.landmarks
        self.rev_neigh = src_detector.rev_neigh
        self.weights = src_detector.weights
        self.communities = src_detector.communities
        self.communities_dict = src_detector.communities_dict
        self.m = src_detector.m

    def run(self):
        repeat = True
        from_landmarks = True
        while repeat:
            # step 1 - fit communities
            iterations = self.__fit_communities(from_landmarks)
            if iterations == 0:
                repeat = False
            # step 2 - rebuild the graph: each community is a new node
            new_detector = self.__get_graph_from_communities(from_landmarks)
            self.__deep_copy(new_detector)
            from_landmarks = False
        return self.graph, self.weights
