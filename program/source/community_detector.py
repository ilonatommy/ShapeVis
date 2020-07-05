import numpy as np

from source import graph
from source.graph import Graph
from source import community
from source.community import Community


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
        total_weight = 0
        if from_landmarks:
            tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)
            for node in self.graph.nodes:
                neighbour = self.graph.adjacency_dict[str(node)]
                for neigh in neighbour:
                    total_weight += tmp_community.get_landmarks_weight_between(neigh, node)
        else:
            total_weight = np.sum(self.weights)
        return total_weight / 2

    def __check_modularity_gain(self, node, dest_community: Community, from_landmarks):
        dest_community_cpy = dest_community.deep_copy()
        dest_community_cpy.nodes.append(node)
        w_inside, w_total = dest_community_cpy.get_community_weights(from_landmarks)
        node_neighbours = self.graph.adjacency_dict[str(node)]
        k_i_in = dest_community_cpy.get_weights_between_node_and_community(node_neighbours, node, from_landmarks)
        k_i = dest_community_cpy.get_weights_of_incident_nodes(node_neighbours, node, from_landmarks)
        d_Q = (k_i_in - (2*w_total * k_i)/(2 * self.m)) / (2 * self.m) # czy na pewno taki wzor?
        return d_Q

    def __shift_node_from_community(self, source_community: Community, dest_community_idx: Community, node, from_landmarks):
        dest_community = self.communities[dest_community_idx]

        source_community.nodes = [n for n in source_community.nodes if (list(n) != node).any()]
        source_community.weights_inside, source_community.weights_total = \
            source_community.get_community_weights(from_landmarks)

        dest_community.nodes.append(node)
        dest_community.weights_inside, dest_community.weights_total = \
            dest_community.get_community_weights(from_landmarks)
        self.communities_dict[(str(node))] = dest_community_idx

    def fit_communities(self, from_landmarks):
        repeat = True
        was_modularity_improved = False
        while repeat:
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
                if len(d_Q) > 0:
                    max_dQ = max(d_Q)
                else:
                    continue

                if max_dQ > 0:
                    dest_community_idx = self.communities_dict[str(neigh)]
                    if dest_community_idx == src_community_idx:
                        continue

                    repeat = True
                    was_modularity_improved = True
                    neigh = neighbours[d_Q.index(max_dQ)]
                    self.__shift_node_from_community(src_community, dest_community_idx, node, from_landmarks)
        
        return was_modularity_improved

    def __get_graph_from_communities(self, from_landmarks):
        communities_number = len(self.communities)
        communities = []
        for i in range(communities_number):
            communities.append(np.array([i]))
        new_graph = Graph(communities)
        new_weights = np.zeros((communities_number, communities_number))
        tmp_community = Community(self.graph, self.landmarks, self.rev_neigh, self.weights)

        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                # if they are from different communities:
                c1_idx = self.communities_dict[str(n1)]
                c2_idx = self.communities_dict[str(n2)]
                if c1_idx != c2_idx:
                    # and are neighbours for each other:
                    if self.graph.is_node2_neighbor_of_node1(n1, n2):
                        # new nodes made from these communities will be neighbours:
                        new_graph.adjacency_dict[str(np.array([c1_idx]))].append(np.array([c2_idx]))
                        new_graph.adjacency_dict[str(np.array([c2_idx]))].append(np.array([c1_idx]))
                        # find weights between n1 and n2
                        if from_landmarks:
                            weight = tmp_community.get_landmarks_weight_between(n1, n2)
                        else:
                            weight = tmp_community.get_weights_between(n1[0], n2[0])
                        new_weights[c1_idx][c2_idx] += weight
                        new_weights[c2_idx][c1_idx] += weight
        new_weights /= 2
        return CommunityDetector(new_graph, self.landmarks, self.rev_neigh, new_weights, from_landmarks=False)

    def __deep_copy(self, src_detector):
        self.graph = src_detector.graph
        self.landmarks = src_detector.landmarks.copy()
        self.rev_neigh = src_detector.rev_neigh.copy()
        self.weights = src_detector.weights.copy()
        self.communities = src_detector.communities.copy()
        self.communities_dict = src_detector.communities_dict.copy()
        self.m = src_detector.m

    def run(self):
        repeat = True #zmienić potem na True
        from_landmarks = True
        while repeat:
            # step 1 - fit communities
            repeat = self.fit_communities(from_landmarks)
            # step 2 - rebuild the graph: each community is a new node
            new_detector = self.__get_graph_from_communities(from_landmarks)
            self.__deep_copy(new_detector)
            from_landmarks = False
        return self.graph, self.weights
