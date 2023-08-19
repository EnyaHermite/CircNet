import numpy as np

class MyAnchor:
    def __init__(self, anchor_delta, anchor_range):
        self.delta = anchor_delta
        self.range = anchor_range
    def init_anchors(self):
        seeds = []
        for key in self.delta.keys():
            seed_key = np.arange(self.range[key][0]+self.delta[key]/2, self.range[key][1], self.delta[key])
            seeds.append(seed_key)
        tuple_cart = np.meshgrid(*seeds)
        tuple_cart_1D = [np.reshape(item,[-1]) for item in tuple_cart]
        anchor_centers = np.stack(tuple_cart_1D, axis=1)
        num_anchors = anchor_centers.shape[0]
        return anchor_centers, num_anchors
