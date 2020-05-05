from __future__ import division

from algo_comparer import AlgoComparer
from data_processor import DataProcessor
from mainfold_landmarker import MainfoldLandmarker


def main():
    compare = False

    data_proc = DataProcessor()
    data_proc.load_mnist()

    landmarker = MainfoldLandmarker(data_proc, 4)
    landmarker.create_knn_graph()

    if compare:
        algo_comparer = AlgoComparer("TSNE")
        algo_comparer.compare(data_proc)


if __name__ == '__main__':
    main()
