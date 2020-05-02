from __future__ import division

from algo_comparer import AlgoComparer
from data_processor import DataProcessor


def main():
    data_proc = DataProcessor()
    data_proc.load_mnist()

    algo_comparer = AlgoComparer("TSNE")
    algo_comparer.compare(data_proc)


if __name__ == '__main__':
    main()
