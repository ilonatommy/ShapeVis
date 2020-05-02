from __future__ import division

import numpy as np


class ManifoldLandmarker:
    def __init__(self, data_proc, distance):
        self.mins =  np.amin(data_proc.data, axis=0)# min values for consecutive dimensions: 0, 1..
        self.distance = distance

    def __sample(self, points, dim, element_dims, results):
        if len(points.shape) > 1:
            for i in range(points.shape[dim]):
                self.__sample(points[i], dim+1, element_dims, results)
        else:
            sample = True
            for point in points:
                sampled_val = (point - self.mins[dim]) / self.distance
                if int(sampled_val) != sampled_val:
                    sample = False
                    break
            if sample:
                results.append(points)
        return np.array(results)

    def __uniform_sampling(self, data_proc):
        result = []
        result = self.__sample(data_proc.data, 0, len(data_proc.data.shape), result)
        print(result)

    def create_knn_graph(self, data_proc):
        self.__uniform_sampling(data_proc)