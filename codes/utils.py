import numpy as np
import tensorflow as tf


class MyUtils:

    def __init__(self, dim, anchor_type):
        self.dim = dim
        self.anchor_type=anchor_type

    def rad2deg(self, radians):
        degrees = radians*180/np.pi
        return degrees

    def deg2rad(self, degrees):
        radians = degrees*np.pi/180
        return radians

    @staticmethod
    def unique(triangles, axis=None):
        return tf.numpy_function(func=np.unique, inp=[triangles, True, True, True, axis],
                                 Tout=[tf.int32, tf.int64, tf.int64, tf.int64])

    @staticmethod
    def argsort_array2_by_rows(array, num_rows, num_cols):
        sort_index = tf.range(num_rows)
        for col in tf.range(num_cols-1, -1, -1):
            idx = tf.argsort(array[:,col])
            array = tf.gather(array, idx)
            sort_index = tf.gather(sort_index, idx)
        return array, sort_index

    def transform_cart_coordinates(self, coords, radius=None):
        # theta: angle with positive x-axis
        # phi:   angle with xy-plane
        if self.anchor_type!='cart':
            theta = tf.math.atan2(coords[:,1], coords[:,0])  # [-PI, PI] in radians
            if radius is None:
                radius = tf.sqrt(tf.reduce_sum(coords**2, axis=-1))

            if self.dim==3:
                radius2 = tf.sqrt(tf.reduce_sum(coords[:,:2]**2, axis=-1))
                phi = tf.math.atan2(coords[:,2], radius2)     # [-PI/2, PI/2] in radians
                new_coords = tf.stack([theta, phi, radius], axis=1)
            else:
                new_coords = tf.stack([theta, radius], axis=1)
        else:
            new_coords = coords
        return new_coords

    def recover_cart_coordinates(self, coords):
        if self.anchor_type!='cart':
            theta = coords[:,0]     # [-PI, PI] in radians
            radius = coords[:,-1]
            if self.dim==3:
                phi = coords[:,1]   # [-PI/2, PI/2] in radians
                new_coords = tf.stack([radius*tf.cos(phi)*tf.cos(theta),
                                       radius*tf.cos(phi)*tf.sin(theta),
                                       radius*tf.sin(phi)], axis=1)
            else:
                new_coords = tf.stack([radius*tf.cos(theta), radius*tf.sin(theta)], axis=1)
        else:
            new_coords = coords
        return new_coords

    def compute_circumcenter(self, points, triangles):
        if self.dim==2:  # pad points to 3D by adding 0 zetas
            points = tf.concat([points, tf.zeros_like(points[:,:1])],axis=-1)

        A = tf.gather(points, triangles[:,0]) # triangle vertex A
        B = tf.gather(points, triangles[:,1]) # triangle vertex B
        C = tf.gather(points, triangles[:,2]) # triangle vertex C

        # calculate the angles
        l_AB = tf.sqrt(tf.reduce_sum((A-B)**2,axis=-1))
        l_AC = tf.sqrt(tf.reduce_sum((A-C)**2,axis=-1))
        l_BC = tf.sqrt(tf.reduce_sum((B-C)**2,axis=-1))
        dot_A = tf.reduce_sum((B-A)*(C-A),axis=-1)/(l_AB*l_AC)
        dot_B = tf.reduce_sum((A-B)*(C-B),axis=-1)/(l_AB*l_BC)
        dot_C = tf.reduce_sum((A-C)*(B-C),axis=-1)/(l_AC*l_BC)
        dot_A = tf.clip_by_value(dot_A, -1., 1.) # to avoid complex values
        dot_B = tf.clip_by_value(dot_B, -1., 1.)
        dot_C = tf.clip_by_value(dot_C, -1., 1.)
        angle_A = tf.math.acos(dot_A)   # [0, PI] in radians
        angle_B = tf.math.acos(dot_B)
        angle_C = tf.math.acos(dot_C)
        angles = tf.stack([angle_A, angle_B, angle_C], axis=1)

        # calculate the circumcirles
        a, b = (A-C, B-C)
        norm_a = tf.sqrt(tf.reduce_sum(a**2,axis=-1,keepdims=True))
        norm_b = tf.sqrt(tf.reduce_sum(b**2,axis=-1,keepdims=True))
        norm_ab_MINUS = tf.sqrt(tf.reduce_sum((a-b)**2,axis=-1,keepdims=True))
        norm_ab_CROSS = tf.sqrt(tf.reduce_sum((tf.linalg.cross(a,b))**2,axis=-1,keepdims=True))
        center = tf.linalg.cross(((norm_a**2)*b - (norm_b**2)*a), tf.linalg.cross(a,b))/(2*norm_ab_CROSS**2+np.finfo(float).eps) + C
        radius = norm_a*norm_b*norm_ab_MINUS/(2*norm_ab_CROSS+np.finfo(float).eps)
        circles = tf.concat([center[:,:self.dim], radius, angles], axis=-1)
        return circles
