from __future__ import division

from source.algo_comparer import AlgoComparer
from source.data_processor import DataProcessor
from source.witness_complex import WitnessComplexGraphBuilder
from source.graph import Graph

def main():
    compare = False

    data_proc = DataProcessor()
    data_proc.load_mnist()

    graph_builder = WitnessComplexGraphBuilder(data_proc, 4)
    graph_builder.build_knn()
    graph_builder.build_augmented_knn()
    graph = graph_builder.get_graph()

    if compare:
        algo_comparer = AlgoComparer("TSNE")
        algo_comparer.compare(data_proc)


if __name__ == '__main__':
    main()
