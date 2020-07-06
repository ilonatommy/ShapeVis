import numpy as np

import graph
from graph import Graph


class Community:
    def __init__(self, graph: graph.Graph, landmarks, rev_neigh, landmark_weights):
        self.landmarks = landmarks
        self.rev_neigh = rev_neigh
        self.graph = graph
        weight_matrix_size = len(self.graph.nodes)
        self.weights = np.zeros((weight_matrix_size, weight_matrix_size))

        self.weights_total = 0
        self.weights_inside = 0
        self.nodes = []

        for n1 in graph.nodes:
            for n2 in graph.nodes:
                try:
                    w_n1_idx = self.landmarks[self.rev_neigh[n1]]
                except:
                    w_n1_idx = self.landmarks[n1]
                try:
                    w_n2_idx = self.landmarks[self.rev_neigh[n2]]
                except:
                    w_n2_idx = self.landmarks[n2]
                if n1 is not n2:
                    self.weights[n1][n2] = landmark_weights[w_n1_idx][w_n2_idx]
                    self.weights[n2][n1] = landmark_weights[w_n1_idx][w_n2_idx]

    def get_weights_between(self, node1, node2):
        return self.weights[node1][node2]

    def __get_weight_inside_community(self):
        total_sum = 0
        # if we have only 1 node in community, so no edges then return 0
        if len(self.nodes) < 2:
            return total_sum
        for node1 in self.nodes:
            for node2 in self.nodes:
                # if the nodes are connected then check what is the weight of the connection:
                if self.graph.is_node2_neighbor_of_node1(node1, node2, False):
                    weight = self.get_weights_between(node1, node2)
                    total_sum += weight
        # for 2 nodes we have 1 weight, which is added twice, but we need it only once
        return total_sum / 2

    def get_community_weights(self):
        # the amount we will get is: tmp_weight = 2*w_inside_comm + 1*incident_weights
        tmp_weight = 0
        for node in self.nodes:
            neighbours = self.graph.adjacency_dict[node]
            for neigh in neighbours:
                tmp_weight += self.get_weights_between(node, neigh)
        weights_inside = self.__get_weight_inside_community()
        weights_total = tmp_weight - weights_inside
        return weights_inside, weights_total

    def get_weights_between_node_and_community(self, neighbours, node_i):  # returns k_i,in
        total_weight = 0
        coeff = 1
        intersection_list = list(set(self.nodes) & set(neighbours))
        for community_neigh in intersection_list:
            total_weight += self.get_weights_between(node_i, community_neigh)
        return total_weight * coeff

    def get_weights_of_incident_nodes(self, neighbours, node_i):
        total_weight = 0
        for neigh in neighbours:
            total_weight += self.get_weights_between(node_i, neigh)
        return total_weight

    def deep_copy(self):
        community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        community.nodes = list.copy(self.nodes)
        community.weights_total = self.weights_total
        community.weights_inside = self.weights_inside
        return community


class CommunityDetector:
    def __init__(self, graph: graph.Graph, landmarks, rev_neigh, weights):
        self.graph = graph
        self.landmarks = landmarks
        self.rev_neigh = rev_neigh
        # weights is always a symmetric matrix
        self.weights = weights
        self.communities = []
        self.communities_dict = {}
        self.m = self.__get_accumulated_network_weights()
        # in initial partition there are as many communities as there are nodes:
        community_idx = 0
        for node in self.graph.nodes:
            new_community = Community(graph, landmarks, rev_neigh, weights)
            new_community.nodes.append(node)
            self.communities.append(new_community)
            # dict stores info in which community to look for a node: complexity of finding community - O(2)
            self.communities_dict[node] = community_idx
            community_idx += 1

    def __get_accumulated_network_weights(self):
        return np.sum(self.weights) / 2

    def __check_modularity_gain(self, node, dest_community: Community):
        dest_community_cpy = dest_community.deep_copy()
        dest_community_cpy.nodes.append(node)
        w_inside, w_total = dest_community_cpy.get_community_weights()
        node_neighbours = self.graph.adjacency_dict[node]
        k_i_in = dest_community_cpy.get_weights_between_node_and_community(node_neighbours, node)
        k_i = dest_community_cpy.get_weights_of_incident_nodes(node_neighbours, node)
        d_Q = (k_i_in - (2 * w_total * k_i) / (2 * self.m)) / (2 * self.m)  # czy na pewno taki wzor?
        return d_Q

    @staticmethod
    def __shift_node_from_community(source_community: Community, dest_community: Community, node):
        source_community.nodes = [n for n in source_community.nodes if n != node]
        source_community.weights_inside, source_community.weights_total = \
            source_community.get_community_weights()
        dest_community.nodes.append(node)
        dest_community.weights_inside, dest_community.weights_total = \
            dest_community.get_community_weights()

    def __get_graph_from_communities(self):
        communities_number = len(self.communities)
        print("communities dict after shift", self.communities_dict)
        new_graph = Graph(list(range(communities_number)), False)
        new_weights = np.zeros((communities_number, communities_number))
        tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        print("communities_number", communities_number)

        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                # if they are from different communities:
                c1_idx = self.communities_dict[n1]
                c2_idx = self.communities_dict[n2]
                # print("c1: ", c1_idx, "c2: ", c2_idx)
                if c1_idx != c2_idx:
                    # and are neighbours for each other:
                    if self.graph.is_node2_neighbor_of_node1(n1, n2, False):
                        # new nodes made from these communities will be neighbours:
                        try:
                            if c2_idx not in new_graph.adjacency_dict[c1_idx]:
                                new_graph.adjacency_dict[c1_idx].append(c2_idx)
                        except:
                            new_graph.adjacency_dict[c1_idx] = [c2_idx]
                        try:
                            if c1_idx not in new_graph.adjacency_dict[c2_idx]:
                                new_graph.adjacency_dict[c2_idx].append(c1_idx)
                        except:
                            new_graph.adjacency_dict[c2_idx] = [c1_idx]
                        # find weights between n1 and n2
                        weight = tmp_community.get_weights_between(n1, n2)
                        new_weights[c1_idx][c2_idx] += weight
                        new_weights[c2_idx][c1_idx] += weight
        new_weights /= 2
        return CommunityDetector(new_graph, self.landmarks, self.rev_neigh, new_weights)

    def __deep_copy(self, src_detector):
        self.graph.nodes = src_detector.graph.nodes
        self.graph.adjacency_dict = src_detector.graph.adjacency_dict
        self.landmarks = src_detector.landmarks.copy()
        self.rev_neigh = src_detector.rev_neigh.copy()
        self.weights = src_detector.weights.copy()
        self.communities = src_detector.communities.copy()
        self.communities_dict = src_detector.communities_dict.copy()
        self.m = src_detector.m

    def fit_communities_for_node(self, node):
        changes = 0
        src_community_idx = self.communities_dict[node]
        src_community = self.communities[src_community_idx]
        neighbours = self.graph.adjacency_dict[node]
        d_Q = []
        for neigh in neighbours:
            community_idx = self.communities_dict[neigh]
            neighbour_community = self.communities[community_idx]
            d_Q.append(self.__check_modularity_gain(node, neighbour_community))
        max_dQ = max(d_Q)
        if max_dQ > 0:
            neigh = neighbours[d_Q.index(max_dQ)]
            community_idx = self.communities_dict[neigh]
            dest_community = self.communities[community_idx]
            if dest_community != src_community:
                print("Node with max d_Q =", node, "is shifted to community of idx: ", community_idx)
                changes += 1
                self.__shift_node_from_community(src_community, dest_community, node)
                # if the shift resulted in emptying the source community then let's get rid of it totally:
                if not src_community.nodes:
                    self.communities = [comm for comm in self.communities if comm != src_community]
                    self.communities_dict[node] = community_idx
                    # if we did not just get rid of the last community (by indexing) then we need to re-numerate idxs:
                    if src_community_idx is not len(self.communities_dict) - 1:
                        print("Renumeration begining from community idx ", src_community_idx)
                        for key in self.communities_dict:
                            if self.communities_dict[key] > src_community_idx:
                                self.communities_dict[key] -= 1
        return changes

    def run(self):
        repeat = True  # zmienić potem na True
        while repeat:
            repeat = False #USUNAC
            per_iteration_changes_cnt = 0
            print("Nodes: ", self.graph.nodes, "adjac.dict.", self.graph.adjacency_dict)
            for node in reversed(self.graph.nodes):
                per_node_changes_cnt = 0
                # step 1 - fit communities
                per_node_changes_cnt += self.fit_communities_for_node(node)
                # step 2 - rebuild the graph after each change: each community is a new node
                if per_node_changes_cnt != 0:
                    new_detector = self.__get_graph_from_communities()
                    self.__deep_copy(new_detector)
                    print("New graph after change; nodes: ", self.graph.nodes, "adjac.dict.", self.graph.adjacency_dict)
                per_iteration_changes_cnt += per_node_changes_cnt
            # if nothing changed through an iteration -> STOP
            if per_iteration_changes_cnt == 0:
                repeat = False
        return self.graph, self.weights
