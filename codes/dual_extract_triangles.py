import numpy as np
import tensorflow as tf
from utils import MyUtils


class MyEvaluate(MyUtils):

    def __init__(self, anchor_delta, anchor_centers, num_preds_per_anchor=2, anchor_type='other',
                 thresholds=None):
        self.dim = anchor_centers.shape[1]
        super(MyEvaluate, self).__init__(dim=self.dim, anchor_type=anchor_type)
        self.anchor_delta = np.asarray(anchor_delta, 'float32')
        self.anchor_centers = np.asarray(anchor_centers, 'float32')
        self.T = tf.shape(anchor_centers)[0]
        self.S = num_preds_per_anchor
        self.anchor_type = anchor_type
        self.thresholds = thresholds

    @tf.function
    def get_predictions(self, pred_offsets, pred_logits, points, knn_indices, knn_scale):

        pred_confidences = tf.nn.sigmoid(pred_logits)
        anchor_centers_3D = tf.expand_dims(self.anchor_centers, axis=0)
        anchor_centers_4D = tf.expand_dims(anchor_centers_3D, axis=2)
        points_4D = tf.expand_dims(tf.expand_dims(points, axis=1),axis=2)
        pred_centers = pred_offsets*tf.reshape(self.anchor_delta, [1,1,1,self.dim]) + anchor_centers_4D
        pred_centers = tf.reshape(pred_centers, [-1,self.dim])
        pred_cart_centers = self.recover_cart_coordinates(pred_centers)
        pred_cart_centers = tf.reshape(pred_cart_centers, [-1, self.T, self.S, self.dim])
        pred_circumcenters = pred_cart_centers + points_4D   # local to global
        pred_circumcenters = tf.math.multiply(pred_circumcenters, knn_scale[:, :, None, None])
        points = tf.math.multiply(points, knn_scale)
        pred_confidences = tf.stack([pred_confidences, pred_confidences], axis=2)
        pred_triangles, \
        pred_confidences = self.extract_triangles(pred_circumcenters, pred_confidences, points, knn_indices)

        return pred_triangles, pred_confidences

    def extract_triangles(self, pred_circumcenters, confidences, points, knn_indices):
        pred_triangles = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)
        pred_confidences = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        pred_max_dist = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        knn_points = tf.gather(points, knn_indices) # shape=[num_points, knn, dim]
        for s in range(self.S):
            pos_indices = tf.cast(tf.where(confidences[...,s]>self.thresholds['confidence']), dtype=tf.int32)
            if tf.shape(pos_indices)[0]==0:
                continue
            pos_centers = tf.gather_nd(pred_circumcenters[...,s,:], pos_indices)
            pos_delta = tf.gather(knn_points, pos_indices[:,0]) - tf.expand_dims(pos_centers, axis=1)
            pos_radius = tf.sqrt(tf.reduce_sum(pos_delta**2, axis=-1))
            pos_radius_delta = tf.abs(pos_radius - pos_radius[...,:1])
            tri_radius_delta, tri_pred = tf.math.top_k(-pos_radius_delta, k=3, sorted=True)  # ascending order
            tri_radius_delta = -tri_radius_delta
            point_indices = tf.tile(tf.reshape(pos_indices[:,0],[-1,1,1]),[1,3,1])
            tri_pred_indices = tf.concat([point_indices, tf.expand_dims(tri_pred[:,:3], axis=2)], axis=2)
            pos_triangles = tf.gather_nd(knn_indices, tri_pred_indices)
            pred_triangles= pred_triangles.write(s, pos_triangles)
            pred_confidences = pred_confidences.write(s, tf.gather_nd(confidences[...,s], pos_indices))
            pred_max_dist = pred_max_dist.write(s, tri_radius_delta[:,2])

        pred_triangles = pred_triangles.concat()
        pred_confidences = pred_confidences.concat()
        pred_max_dist = pred_max_dist.concat()

        sort_idx = tf.argsort(pred_confidences, axis=0, direction='DESCENDING')
        pred_triangles = tf.gather(pred_triangles, sort_idx) # sort triangles with descending confidence
        pred_confidences = tf.gather(pred_confidences, sort_idx)

        pred_triangles = tf.sort(pred_triangles, axis=1)  # sort vertices of each triangle
        pred_triangles, uni_indices, _, _ = self.unique(pred_triangles, axis=0)
        pred_confidences = tf.gather(pred_confidences, uni_indices)
        return pred_triangles, pred_confidences
