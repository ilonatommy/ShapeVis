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

    def get_weight_between(self, node1, node2):
        try:
            w_n1_idx = self.landmarks[str(node1)]
        except:
            w_n1_idx = self.landmarks[self.rev_neigh[str(node1)]]
        try:
            w_n2_idx = self.landmarks[str(node2)]
        except:
            w_n2_idx = self.landmarks[self.rev_neigh[str(node2)]]
        return self.weights[w_n1_idx][w_n2_idx]

    def __get_weight_inside_community(self):
        total_sum = 0
        # if we have only 1 node in community, so no edges then return 0
        if len(self.nodes) < 2:
            return total_sum
        for node1 in self.nodes:
            for node2 in self.nodes:
                if not Graph.are_equal_nodes(node1, node2):
                    weight = self.get_weight_between(node1, node2)
                    total_sum += weight
        # for 2 nodes we have 1 weight, which is added twice, but we need it only once
        return total_sum / 2

    def get_community_weights(self):
        # the amount we will get is: tmp_weight = 2*w_inside_comm + 1*incident_weights
        tmp_weight = 0
        for node in self.nodes:
            neighbours = self.graph.adjacency_dict[str(node)]
            for neigh in neighbours:
                tmp_weight += self.get_weight_between(node, neigh)
        weights_inside = self.__get_weight_inside_community()
        weights_total = tmp_weight - weights_inside
        return weights_inside, weights_total

    def get_weights_between_node_and_community(self, neighbours, node_i): # returns k_i,in
        total_weight = 0
        intersection_list = list(set(self.nodes) & set(neighbours))
        for community_neigh in intersection_list:
            total_weight += self.get_weight_between(node_i, community_neigh)
        return total_weight

    def get_weights_of_incident_nodes(self, neighbours, node_i):
        total_weight = 0
        for neigh in neighbours:
            total_weight += self.get_weight_between(node_i, neigh)
        return total_weight

    def deep_copy(self):
        community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        community.nodes = self.nodes
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
            self.communities_dict[(str(node))] = community_idx
            community_idx += 1

    def __get_accumulated_network_weights(self):
        tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
        total_weight = 0
        for node in self.graph.nodes:
            neighbour = self.graph.adjacency_dict[str(node)]
            for neigh in neighbour:
                total_weight += tmp_community.get_weight_between(neigh, node)
        return total_weight / 2

    def __check_modularity_gain(self, node, dest_community: Community):
        dest_community_cpy = dest_community.deep_copy()
        dest_community_cpy.nodes.append(node)
        w_inside, w_total = dest_community_cpy.get_community_weights()
        node_neighbours = self.graph.adjacency_dict[str(node)]
        k_i_in = dest_community_cpy.get_weights_between_node_and_community(node_neighbours, node)
        k_i = dest_community_cpy.get_weights_of_incident_nodes(node_neighbours, node)
        d_Q = (k_i_in - 2*w_total * k_i) / (2 * self.m) # czy na pewno taki wzor?
        return d_Q

    @staticmethod
    def __shift_node_from_community(source_community: Community, dest_community: Community, node):
        source_community.nodes.remove(node)
        source_community.weights_inside, source_community.weights_total = \
            source_community.get_community_weights()
        dest_community.nodes.append(node)
        dest_community.weights_inside, dest_community.weights_total = \
            dest_community.get_community_weights()

    def __fit_communities(self):
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
                    d_Q.append(self.__check_modularity_gain(node, neighbour_community))
                max_dQ = max(d_Q)
                if max_dQ > 0:
                    repeat = True
                    iteration_idx += 1
                    neigh = neighbours[d_Q.index(max_dQ)]
                    community_idx = self.communities_dict[str(neigh)]
                    dest_community = self.communities[community_idx]
                    self.__shift_node_from_community(src_community, dest_community, node)
        return iteration_idx

    def run(self):
        repeat = True
        while(repeat):
            iterations = self.__fit_communities()
            if iterations == 0:
                repeat = False
            # TODO: step 2 - rebuild the graph: each community is a new node