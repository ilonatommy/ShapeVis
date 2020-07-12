import numpy as np

def assert_adjacency_dicts_are_equal(adjacency_dict, expected_adjacency_dict):
    for key in expected_adjacency_dict.keys():
        np.testing.assert_array_equal(adjacency_dict[key], expected_adjacency_dict[key])
