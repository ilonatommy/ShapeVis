from __future__ import division
import time

from source.algo_comparer import AlgoComparer
from source.data_processor import DataProcessor
from source.witness_complex import WitnessComplexGraphBuilder
from source.landmark_selector import LandmarkSelector
from source.random_walker import RandomWalker
from source.community_detector import CommunityDetector
from source.manifold_tearer import ManifoldTearer


def main():
    compare = False
    
    prev_time = time.time()
    data_proc = DataProcessor()
    print("DataProcessor() time: ", time.time() - prev_time)
    
    prev_time = time.time()
    data_proc.load_mnist(limit=50)
    print("load_mnist() time: ", time.time() - prev_time)
    
    prev_time = time.time()
    graph_builder = WitnessComplexGraphBuilder(data_proc, 30)
    print("WitnessComplexGraphBuilder() time: ", time.time() - prev_time)

    prev_time = time.time()
    graph_builder.build_knn(k=3)
    print("build_knn() time: ", time.time() - prev_time)

    prev_time = time.time()
    graph_builder.build_augmented_knn()
    print("build_augmented_knn() time: ", time.time() - prev_time)

    graph = graph_builder.get_graph()

    prev_time = time.time()
    landmark_selector = LandmarkSelector(graph)
    print("LandmarkSelector() time: ", time.time() - prev_time)

    prev_time = time.time()
    landmark_selector.select_landmarks()
    print("select_landmarks() time: ", time.time() - prev_time)

    landmarks = landmark_selector.get_landmarks()
    rev_neigh = landmark_selector.get_rev_neigh()

    prev_time = time.time()
    random_walker = RandomWalker(graph, len(landmarks), 20, 1, 1)
    print("RandomWalker() time: ", time.time() - prev_time)

    prev_time = time.time()
    random_walker.walk(landmarks, rev_neigh)
    print("walk() time: ", time.time() - prev_time)
    
    prev_time = time.time()
    random_walker.calculate_weigths(2)
    print("calculate_weigths() time: ", time.time() - prev_time)
    
    w_matrix = random_walker.get_w_matrix()
    labels = [graph.nodes[landmark]["label"] for landmark in list(landmarks.keys())]

    prev_time = time.time()
    igp_graph, igp_labels = CommunityDetector.detect_communities(w_matrix, labels)
    print("CommunityDetector time: ", time.time() - prev_time)

    prev_time = time.time()
    final_graph = ManifoldTearer.reduce_edges(igp_graph, igp_labels)
    print("Manifold tearing time: ", time.time() - prev_time)

    if compare:
        algo_comparer = AlgoComparer("TSNE")
        algo_comparer.compare(data_proc)


if __name__ == '__main__':
    main()
