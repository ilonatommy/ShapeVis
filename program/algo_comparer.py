from __future__ import division

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class AlgoComparer:
    def __init__(self, algo_type):
        self.algo_type = algo_type
        if algo_type == "TSNE":
            self.transformation = TSNE(n_components=2)
        else:
            print("Transformation not available. Unable to compare.")

    def visualise_transformed(self, points_transformed, colors, labels):
        points_transformed_t = points_transformed.T

        fig = plt.figure()
        ax = fig.add_subplot()
        scatter = ax.scatter(points_transformed_t[0], points_transformed_t[1], c=colors, cmap=plt.cm.coolwarm)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.legend(handles=scatter.legend_elements()[0], labels=labels)
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.show()

    def compare(self, data_processor):
        self.transformed_data = self.transformation.fit_transform(data_processor.data)
        self.visualise_transformed(self.transformed_data, data_processor.labels, data_processor.names)
