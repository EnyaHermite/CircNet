import numpy as np
import tensorflow as tf
from utils import MyUtils


class MyLoss(MyUtils):

    def __init__(self, anchor_delta, anchor_centers, num_preds_per_anchor=2, anchor_type='other', alpha1=1.,
                 alpha2=1., thresholds=None, negative_ratio=20):
        self.dim = anchor_centers.shape[1]
        super(MyLoss, self).__init__(dim=self.dim, anchor_type=anchor_type)
        self.anchor_delta = np.asarray(anchor_delta, 'float32')
        self.anchor_centers = np.asarray(anchor_centers, 'float32')
        self.T = tf.shape(anchor_centers)[0]
        self.S = num_preds_per_anchor
        self.anchor_type = anchor_type
        self.alpha1 = alpha1  
        self.alpha2 = alpha2
        self.thresholds = thresholds
        self.hard_negative_ratio = negative_ratio
        self.Beta = 10000
        assert(self.Beta>self.T)

    def hard_negative_mining(self, gt_labels, pred_logits):
        pos_pred = tf.gather_nd(pred_logits, tf.where(gt_labels==1))  
        neg_pred = tf.gather_nd(pred_logits, tf.where(gt_labels==0))  
        neg_pred = tf.sort(neg_pred, direction='DESCENDING') 

        N_pos = tf.reduce_sum(tf.cast(gt_labels==1, tf.int32))
        N_neg = tf.reduce_sum(tf.cast(gt_labels==0, tf.int32))
        
        N_neg = tf.minimum(self.hard_negative_ratio*N_pos, N_neg)
        neg_pred = neg_pred[:N_neg]

        loss_cls_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_pred), logits=pos_pred))
        loss_cls_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_pred), logits=neg_pred))
        loss_cls = loss_cls_pos + loss_cls_neg
        return loss_cls, loss_cls_pos, loss_cls_neg

    def build_loss(self, pred_offsets, pred_logits, points, knn_indices, knn_scale, gt_triangles):
        
        sample_indices = tf.squeeze(knn_indices[:, :1])
        G2L_point_indices_map= tf.scatter_nd(tf.expand_dims(sample_indices, axis=-1), updates=tf.range(tf.shape(sample_indices)[0]), shape=tf.shape(points[:,0]))
        
        anchor_centers_3D = tf.expand_dims(self.anchor_centers, axis=0)
        
        anchor_centers_4D = tf.expand_dims(anchor_centers_3D, axis=2)
        points_4D = tf.expand_dims(tf.expand_dims(points, axis=1),axis=2)

        patch_points_4D = tf.gather(points_4D, sample_indices)
        
        pred_centers = pred_offsets*tf.reshape(self.anchor_delta, [1,1,1,self.dim]) + anchor_centers_4D
        pred_centers = tf.reshape(pred_centers, [-1,self.dim])
        pred_cart_centers = self.recover_cart_coordinates(pred_centers)
        pred_cart_centers = tf.reshape(pred_cart_centers, [-1, self.T, self.S, self.dim])
        pred_circumcenters = pred_cart_centers + patch_points_4D  
        pred_circumcenters = tf.math.multiply(pred_circumcenters, tf.gather(knn_scale[:, :, None, None], sample_indices))
        
        point_mask = tf.scatter_nd(tf.expand_dims(sample_indices, axis=-1), updates=tf.ones_like(sample_indices), shape=tf.shape(points[:,0]))
        gt_triangle_mask_1d = tf.gather(point_mask, tf.reshape(gt_triangles, [-1])) 
        gt_triangle_mask = tf.reshape(gt_triangle_mask_1d, [-1, 3])
        

        points = tf.math.multiply(points, knn_scale)
        gt_circles = self.compute_circumcenter(points, gt_triangles) 

        valid_triangle_mask = tf.logical_and(gt_circles[:,-3:]>self.thresholds['min_angle'],
                                             gt_circles[:,-3:]<self.thresholds['max_angle'])
        valid_triangle_mask = tf.reduce_all(valid_triangle_mask, axis=-1, keepdims=True)

        matched_offsets_gt = tf.TensorArray(tf.float32, size=3, dynamic_size=False, infer_shape=False)
        matched_offsets_indices = tf.TensorArray(tf.int32, size=3, dynamic_size=False, infer_shape=False)
         
        for vid in range(3):
            sample_vertex_mask = tf.cast(gt_triangle_mask[:,vid], tf.bool)
            gt_loc_coords = tf.math.divide((tf.boolean_mask(gt_circles[:,:self.dim], mask=sample_vertex_mask) - 
                                            tf.gather(points, tf.boolean_mask(gt_triangles[:,vid], mask=sample_vertex_mask))), 
                                           tf.gather(knn_scale, tf.boolean_mask(gt_triangles[:,vid], mask=sample_vertex_mask)))
            gt_centers = self.transform_cart_coordinates(gt_loc_coords, radius=None) 
            gt_offsets = (tf.expand_dims(gt_centers,axis=1) - anchor_centers_3D)\
                         /tf.reshape(self.anchor_delta, [1,1,self.dim]) 
            is_matched = tf.reduce_all(tf.math.logical_and(gt_offsets>-0.5, gt_offsets<0.5), axis=-1)  
            is_matched = tf.logical_and(tf.boolean_mask(valid_triangle_mask, mask=sample_vertex_mask), is_matched) 

            gt_indices = tf.cast(tf.where(is_matched), dtype=tf.int32)
            triangle_indices, anchor_indices = gt_indices[:,0], gt_indices[:,1]
            global_point_indices = tf.gather(tf.boolean_mask(gt_triangles[:,vid], mask=sample_vertex_mask), triangle_indices)
            local_point_indices = tf.gather(G2L_point_indices_map, global_point_indices)
            pred_indices = tf.stack([local_point_indices, anchor_indices], axis=1)

            gt_offsets_circles = tf.concat([tf.gather_nd(gt_offsets, gt_indices),
                                            tf.gather(tf.boolean_mask(gt_circles, mask=sample_vertex_mask), gt_indices[:,0])], axis=1)
            matched_offsets_gt = matched_offsets_gt.write(vid, gt_offsets_circles)
            matched_offsets_indices = matched_offsets_indices.write(vid, pred_indices)

        matched_offsets_gt = matched_offsets_gt.concat()
        matched_offsets_indices = matched_offsets_indices.concat()
        matched_offsets_indices = tf.cast(matched_offsets_indices, tf.int64)
        matched_offsets_indices = matched_offsets_indices[:,0]*self.Beta + matched_offsets_indices[:,1]

        sort_index = tf.argsort(matched_offsets_indices)
        matched_offsets_gt = tf.gather(matched_offsets_gt, sort_index)
        matched_offsets_indices = tf.gather(matched_offsets_indices, sort_index)

        uni_values, uni_inverse, uni_counts = tf.unique_with_counts(matched_offsets_indices)
        point_indices = tf.cast(uni_values/self.Beta, tf.int32)
        anchor_indices = tf.cast(tf.math.mod(uni_values, self.Beta), tf.int32)
        uni_values = tf.stack([point_indices, anchor_indices], axis=1)
        inverse_counts = tf.gather(uni_counts, uni_inverse)

        loss_loc = 0.
        h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        for s in range(self.S):
            gt_indices = tf.where(inverse_counts==(s+1))
            gt_offsets_circles = tf.gather_nd(matched_offsets_gt, gt_indices)
            gt_offsets_circles = tf.tile(gt_offsets_circles[:,:self.dim], [1, self.S-s])
            gt_offsets_circles = tf.reshape(gt_offsets_circles,[-1, self.dim])

            pred_indices_s = tf.gather_nd(uni_values, tf.where(uni_counts==(s+1)))
            pred_offsets_s = tf.gather_nd(pred_offsets, pred_indices_s)
            pred_offsets_s = tf.reshape(pred_offsets_s, [-1, self.dim])
            loss_loc += h(gt_offsets_circles[:,:self.dim], pred_offsets_s)

        updates = tf.ones(tf.shape(uni_values[:,0]), dtype=tf.float32)
        gt_labels = tf.scatter_nd(uni_values, updates=updates, shape=tf.shape(pred_logits))

        skip_indices = tf.gather_nd(uni_values, tf.where(uni_counts>self.S))
        updates = tf.ones(tf.shape(skip_indices[:,0]), dtype=tf.bool)
        skip_mask = tf.scatter_nd(skip_indices, updates=updates, shape=tf.shape(gt_labels))
        skip_mask = tf.cast(tf.logical_not(skip_mask), tf.float32)
        gt_labels_with_skip = skip_mask*gt_labels
        pred_logits_with_skip = skip_mask*pred_logits
        loss_cls, loss_cls_p, loss_cls_n = self.hard_negative_mining(gt_labels_with_skip, pred_logits_with_skip)

        N_pos = tf.reduce_sum(gt_labels_with_skip)*self.S    
        loss_loc = loss_loc/tf.cast(N_pos, dtype=tf.float32)
        

        gt_triangles = tf.gather_nd(gt_triangles, tf.where(valid_triangle_mask[:,0]))
        pred_confidences = tf.nn.sigmoid(pred_logits)
        pred_confidences = tf.stack([pred_confidences, pred_confidences], axis=2) 
        pred_triangles, pred_triangle_confidences = self.extract_triangles(pred_circumcenters, pred_confidences, points, knn_indices)
        
        if tf.reduce_all(pred_triangles == tf.constant([-1,-1,-1])):
            tri_acc, tri_miou, num_correct_pred_triangles = 0.0, 0.0, 0
        else:
            tri_acc, tri_miou, correct_pred_triangles = self.evaluate_triangle_accuracy(pred_triangles, gt_triangles)
            num_correct_pred_triangles = tf.shape(correct_pred_triangles[:,0])

        loss = self.alpha1*loss_loc + self.alpha2*loss_cls 

        return loss, self.alpha1*loss_loc, self.alpha2*loss_cls, self.alpha2*loss_cls_p, self.alpha2*loss_cls_n, tri_acc, tri_miou, num_correct_pred_triangles
    

    def extract_triangles(self, pred_circumcenters, confidences, points, knn_indices):
        pred_triangles = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)
        pred_confidences = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        knn_points = tf.gather(points, knn_indices) 
        for s in range(self.S):
            pos_indices = tf.cast(tf.where(confidences[...,s]>self.thresholds['confidence']), dtype=tf.int32)
            if tf.shape(pos_indices)[0]==0:
                continue
            pos_centers = tf.gather_nd(pred_circumcenters[...,s,:], pos_indices) 
            pos_delta = tf.gather(knn_points, pos_indices[:,0]) - tf.expand_dims(pos_centers, axis=1)
            pos_radius = tf.sqrt(tf.reduce_sum(pos_delta**2, axis=-1))
            pos_radius_delta = tf.abs(pos_radius - pos_radius[...,:1])
            tri_radius_delta, tri_pred = tf.math.top_k(-pos_radius_delta, k=3, sorted=True)  
            tri_radius_delta = -tri_radius_delta
            point_indices = tf.tile(tf.reshape(pos_indices[:,0],[-1,1,1]),[1,3,1])
            tri_pred_indices = tf.concat([point_indices, tf.expand_dims(tri_pred[:,:3], axis=2)], axis=2)
            pos_triangles = tf.gather_nd(knn_indices, tri_pred_indices) 
            pred_triangles= pred_triangles.write(s, pos_triangles)
            pred_confidences = pred_confidences.write(s, tf.gather_nd(confidences[...,s], pos_indices))

        if not pred_triangles.size() == 0:
            pred_triangles = pred_triangles.concat()
            pred_confidences = pred_confidences.concat()

            sort_idx = tf.argsort(pred_confidences, axis=0, direction='DESCENDING')
            pred_triangles = tf.gather(pred_triangles, sort_idx) 
            pred_confidences = tf.gather(pred_confidences, sort_idx)

            pred_triangles = tf.sort(pred_triangles, axis=1)  
            pred_triangles, uni_indices, _, _ = self.unique(pred_triangles, axis=0)
            pred_confidences = tf.gather(pred_confidences, uni_indices)
            return pred_triangles, pred_confidences
        else:
            return tf.constant([-1,-1,-1]), tf.constant(-1.0)

    
    
    def evaluate_triangle_accuracy(self, pred_triangles, gt_triangles):
        gt_triangles = tf.sort(gt_triangles, axis=1)  
        gt_triangles, _, _, _ = self.unique(gt_triangles, axis=0)

        combined_triangles = tf.concat([pred_triangles, gt_triangles], axis=0)
        uni_triangles, _, _, uni_counts = self.unique(combined_triangles, axis=0)
        correct_pred_triangles = tf.gather_nd(uni_triangles, tf.where(uni_counts==2)) 

        num_intersect = tf.reduce_sum(tf.cast(uni_counts==2, dtype=tf.float32))
        num_union = tf.cast(tf.shape(uni_triangles)[0], dtype=tf.float32)
        num_total = tf.cast(tf.shape(gt_triangles)[0], dtype=tf.float32)
        tri_acc = num_intersect/num_total
        tri_miou = num_intersect/num_union

        return tri_acc, tri_miou, correct_pred_triangles

