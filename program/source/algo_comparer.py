from __future__ import division

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import source.helpers as helpers
import networkx as nx
import pandas as pd
from matplotlib.pyplot import figure

class AlgoComparer:
    def __init__(self, algo_type):
        self.algo_type = algo_type
        if algo_type == "TSNE":
            self.transformation = TSNE(n_components=2)
        else:
            print("Transformation not available. Unable to compare.")

    def __calc_dist_between_points(self, set_1, set_2):
        dist = 0
        for coords_1 in set_1:
            for coords_2 in set_2:
                x1 = coords_1[0]
                y1 = coords_1[1]
                x2 = coords_2[0]
                y2 = coords_2[1]
                dist += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

    def __calc_dist_within_classes(self, class_names, classes_strength, classes_coords):
        classes_dist = []
        for class_index in range(len(class_names)):
            class_coords = classes_coords[class_index]
            dist = self.__calc_dist_between_points(class_coords, class_coords)
            dist /= (classes_strength[class_index] * 2)  # razy 2, bo każda odległość będzie liczona dwukrotnie
            classes_dist.append(dist)
        return classes_dist

    def __calc_dist_between_classes(self, class_names, classes_strength, classes_coords):
        classes_dist = []
        for class_1_index in range(len(class_names)):
            dist = 0
            strength = 0
            for class_2_index in range(len(class_names)):
                if class_1_index != class_2_index:
                    class_1_coords = classes_coords[class_1_index]
                    class_2_coords = classes_coords[class_2_index]
                    dist += self.__calc_dist_between_points(class_1_coords, class_2_coords)
                    strength += classes_strength[class_1_index] + classes_strength[class_2_index]
            classes_dist.append(dist / strength)
        return classes_dist

    def __prepare_dist_matrices(self, class_names, labels, transformation_data):
        classes_strength = np.zeros(len(class_names))
        classes_coords = []
        for class_name_idx in range(len(class_names)):
            class_coords = []
            for index in range(len(labels)):
                if str(labels[index]) == str(class_name_idx):
                    classes_strength[class_name_idx] += 1
                    class_coords.append(transformation_data[index])
            classes_coords.append(class_coords)
        return (classes_strength, classes_coords)

    def check_method_quality(self, class_names, class_labels, transformed_data):
        (classes_strength, classes_coords) = self.__prepare_dist_matrices(class_names, class_labels, transformed_data)
        classes_inner_dist = self.__calc_dist_within_classes(class_names, classes_strength, classes_coords)
        classes_outer_dist = self.__calc_dist_between_classes(class_names, classes_strength, classes_coords)
        coeff = sum(classes_inner_dist) / sum(classes_outer_dist)
        print(coeff)

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
        self.check_method_quality(data_processor.names, data_processor.labels, self.transformed_data)
        # self.visualise_transformed(self.transformed_data, list(map(int, data_processor.labels.tolist())),
        # data_processor.names)
