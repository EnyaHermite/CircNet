import tensorflow as tf
import numpy as np


class Augment:

    def __init__(self):
        pass

    @classmethod
    def rotate_point_cloud(cls, xyz, max_angle=2*np.pi, upaxis=3, prob=0.95):
        """ Randomly rotate the point clouds to augment the dataset
            rotation is vertical
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if tf.random.uniform([])<prob:
            angle = tf.random.uniform([1], minval=0, maxval=max_angle)
            if upaxis==1:
                R = cls.rot_x(angle)
            elif upaxis==2:
                R = cls.rot_y(angle)
            elif upaxis==3:
                R = cls.rot_z(angle)
            else:
                raise ValueError('unkown spatial dimension')
            xyz = tf.matmul(xyz, R)
        return xyz

    @classmethod
    def rotate_perturbation_point_cloud(cls, xyz, angle_sigma=0.06, prob=0.95):
        """ Randomly perturb the point clouds by small rotations
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if tf.random.uniform([])<prob:
            angle = tf.random.normal([3], stddev=angle_sigma)
            angle = tf.clip_by_value(angle, -3*angle_sigma, 3*angle_sigma)
            Rx = cls.rot_x([angle[0]])
            Ry = cls.rot_y([angle[1]])
            Rz = cls.rot_z([angle[2]])
            R = tf.matmul(Rz, tf.matmul(Ry, Rx))
            xyz = tf.matmul(xyz, R)
        return xyz

    @classmethod
    def rotate_perturbation_normals(cls, normals, angle_sigma=0.06, prob=0.95):
        """ Randomly perturb the normals by small rotations
            Input:
              Nx3 array, original point cloud normals
            Return:
              Nx3 array, rotated point cloud normals
        """
        if tf.random.uniform([]) < prob:
            angle = tf.random.normal(tf.shape(normals), stddev=angle_sigma)
            angle = tf.clip_by_value(angle, -3*angle_sigma, 3*angle_sigma)
            Rx = cls.rot_x(angle[:,0])
            Ry = cls.rot_y(angle[:,1])
            Rz = cls.rot_z(angle[:,2])
            R = tf.matmul(Rz, tf.matmul(Ry, Rx))
            normals = tf.matmul(tf.expand_dims(normals,axis=1), R)
            normals = tf.squeeze(normals)
        return normals

    @classmethod
    def rotate_point_cloud_by_angle(cls, xyz, angle, upaxis=3):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if upaxis==1:
            R = cls.rot_x(angle)
        elif upaxis==2:
            R = cls.rot_y(angle)
        elif upaxis==3:
            R = cls.rot_z(angle)
        else:
            raise ValueError('unkown spatial dimension')
        rotated_xyz = tf.matmul(xyz, R)
        return rotated_xyz

    @staticmethod
    def jitter_point_cloud(xyz, sigma=0.001, prob=0.95):
        """ Randomly jitter point heights.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        if tf.random.uniform([])<prob:
            noise = tf.random.normal(tf.shape(xyz), stddev=sigma)
            noise = tf.clip_by_value(noise, -3*sigma, 3*sigma)
            xyz += noise
        return xyz

    @staticmethod
    def random_drop_vertex(vertex, face, vertex_label=None, face_label=None, drop_rate=0.1, prob=0.5):
        if vertex_label is not None:
            label = vertex_label
        elif face_label is not None:
            label = face_label
        else:
            label = None

        face_mask = tf.ones([face.shape[0]], dtype=tf.bool)
        if tf.random.uniform([])<prob:
            num_vertices = tf.shape(vertex)[0]
            vertex_mask = tf.cast(tf.random.uniform([num_vertices])<drop_rate,
                                  dtype=tf.int32)  # 1 for mask out, 0 for keep

            face_mask = tf.gather(vertex_mask, face[:,0]) +\
                        tf.gather(vertex_mask, face[:,1]) +\
                        tf.gather(vertex_mask, face[:,2])
            face_mask = (face_mask==0)  # True for keep, False for remove
            valid_facet_indices = tf.where(face_mask)[:,0]
            face = tf.gather(face, valid_facet_indices)

            uni_ids, idx = tf.unique(tf.reshape(face,[-1]))
            face = tf.reshape(idx, [-1, 3])
            vertex = tf.gather(vertex, uni_ids)

            if vertex_label is not None:
                label = tf.gather(vertex_label, uni_ids)
            elif face_label is not None:
                label = tf.gather(face_label, tf.where(face_mask==0))[:,0]

        return vertex, face, label, face_mask

    @staticmethod
    def shift_point_cloud(xyz, sigma=0.1, prob=0.95):
        """ Randomly shift point cloud.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, shifted point clouds
        """
        if tf.random.uniform([])<prob:
            shifts = tf.random.normal([3], stddev=sigma)
            xyz += shifts
        return xyz

    @staticmethod
    def random_scale_point_cloud(xyz, scale_low=0.8, scale_high=1.2, prob=0.95):
        """ Randomly scale the point cloud.
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, scaled point clouds
        """
        if tf.random.uniform([])<prob:
            scales = tf.random.uniform([3], scale_low, scale_high)
            xyz *= scales
        return xyz

    @staticmethod
    def flip_point_cloud(xyz, prob=0.95):
        """ Randomly flip the point cloud.
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, flipped point clouds
        """
        if tf.random.uniform([])<prob:
            x = xyz[:,0]
            y = xyz[:,1]
            z = xyz[:,2]
            if tf.random.uniform([])<0.5:
                x = - x  # flip x-dimension(horizontal)
            else: #if tf.random.uniform([])<0.5:
                y = - y  # flip y-dimension(vertical)
            xyz = tf.stack([x, y, z], axis=1)
        return xyz

    @staticmethod
    def rot_x(angle):
        cosval = tf.cos(angle)
        sinval = tf.sin(angle)
        val0 = tf.zeros_like(cosval)
        val1 = tf.ones_like(cosval)
        R = tf.stack([val1, val0, val0,
                      val0, cosval, -sinval,
                      val0, sinval, cosval], axis=1)
        R = tf.reshape(R, (-1, 3, 3))
        R = tf.squeeze(R)
        return R

    @staticmethod
    def rot_y(angle):
        cosval = tf.cos(angle)
        sinval = tf.sin(angle)
        val0 = tf.zeros_like(cosval)
        val1 = tf.ones_like(cosval)
        R = tf.stack([cosval, val0, sinval,
                      val0, val1, val0,
                      -sinval, val0, cosval], axis=1)
        R = tf.reshape(R, (-1, 3, 3))
        R = tf.squeeze(R)
        return R

    @staticmethod
    def rot_z(angle):
        cosval = tf.cos(angle)
        sinval = tf.sin(angle)
        val0 = tf.zeros_like(cosval)
        val1 = tf.ones_like(cosval)
        R = tf.stack([cosval, -sinval, val0,
                      sinval, cosval, val0,
                      val0, val0, val1], axis=1)
        R = tf.reshape(R, (-1, 3, 3))
        R = tf.squeeze(R)
        return R

def test_rot_x(angle):
    cosval = tf.cos(angle)
    sinval = tf.sin(angle)
    val0 = tf.zeros_like(cosval)
    val1 = tf.ones_like(cosval)
    R = tf.stack([val1, val0, val0,
                  val0, cosval, -sinval,
                  val0, sinval, cosval], axis=1)
    R = tf.reshape(R, (-1, 3, 3))
    R = tf.squeeze(R)
    return R

def test_rot_y(angle):
    cosval = tf.cos(angle)
    sinval = tf.sin(angle)
    val0 = tf.zeros_like(cosval)
    val1 = tf.ones_like(cosval)
    R = tf.stack([cosval, val0, sinval,
                  val0, val1, val0,
                  -sinval, val0, cosval], axis=1)
    R = tf.reshape(R, (-1, 3, 3))
    R = tf.squeeze(R)
    return R

def test_rot_z(angle):
    cosval = tf.cos(angle)
    sinval = tf.sin(angle)
    val0 = tf.zeros_like(cosval)
    val1 = tf.ones_like(cosval)
    R = tf.stack([cosval, -sinval, val0,
                  sinval, cosval, val0,
                  val0, val0, val1], axis=1)
    R = tf.reshape(R, (-1, 3, 3))
    R = tf.squeeze(R)
    return R

if __name__=='__main__':
    raw_normals = tf.random.normal([10,3])
    length = tf.sqrt(tf.reduce_sum(raw_normals**2, axis=-1, keepdims=True))
    normals = raw_normals/length
    length = tf.sqrt(tf.reduce_sum(normals ** 2, axis=-1, keepdims=True))
    print(tf.reduce_min(length), tf.reduce_max(length))

    angle = tf.random.normal([10,3], stddev=0.3)
    Rx = test_rot_z(angle[:,0])
    Ry = test_rot_z(angle[:,1])
    Rz = test_rot_z(angle[:,2])
    R = tf.matmul(Rz, tf.matmul(Ry, Rx))
    print(Rx)
    print(Ry)
    print(Rz)
    R = test_rot_z(angle[:1,-1])
    print(R)
