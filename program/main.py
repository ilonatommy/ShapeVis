from __future__ import division

from algo_comparer import AlgoComparer
from data_processor import DataProcessor
from manifold_landmerker import ManifoldLandmarker


def main():
    compare = True
    dev = True

    data_proc = DataProcessor()
    if dev:
        data_proc.load_artificial_data()
    else:
        data_proc.load_mnist()

    landmarker = ManifoldLandmarker(data_proc, 0.5)
    landmarker.create_knn_graph(data_proc)

    if compare:
        algo_comparer = AlgoComparer("TSNE")
        algo_comparer.compare(data_proc)


if __name__ == '__main__':
    main()
