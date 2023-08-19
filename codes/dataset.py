import tensorflow as tf
from utils import MyUtils


class MyDataset(MyUtils):
    
    def __init__(self, fileList, anchor, dim=3, knn=50, anchor_type='other'):
        super(MyDataset, self).__init__(dim=dim, anchor_type=anchor_type)
        self.dataset = tf.data.Dataset.from_tensor_slices(fileList)
        self.anchor = self.recover_cart_coordinates(tf.constant(anchor, dtype=tf.float32))
        self.knn = knn
        if self.dim==3:
            self.anchor_radii = tf.sqrt(tf.reduce_sum(self.anchor**2,keepdims=True,axis=-1))
            self.anchor_vector = tf.expand_dims(self.anchor/self.anchor_radii,axis=0)

    def shuffle(self, buffer_size=10000):
        self.dataset = self.dataset.shuffle(buffer_size)
        return self

    def map(self, parse_fn, is_training=True):
        self.parse_fn = parse_fn
        self.is_training = is_training
        # self.dataset = tf.data.TFRecordDataset(self.dataset)
        return self

    def batch(self, batch_size, drop_remainder=False):
        new_offset_idx = 0
        points, features, knn_indices, knn_scale, triangles, nv,  mf = [], [], [], [], [], [], []
        # for raw_record in self.dataset.take(-1):
        #     data = self.parse_fn(raw_record.numpy(), dim=self.dim, knn=self.knn, is_training=self.is_training)
        for raw_path in self.dataset:
            data = self.parse_fn(raw_path.numpy(), dim=self.dim, knn=self.knn, is_training=self.is_training)
            points.append(data[0])
            features.append(data[1])
            knn_indices.append(data[2]+new_offset_idx) 
            knn_scale.append(data[3])
            triangles.append(data[4]+new_offset_idx)
            nv.append(data[0].shape[0])
            mf.append(data[4].shape[0])
            new_offset_idx += nv[-1]

            if len(points)==batch_size:
                points = tf.concat(points, axis=0)
                features = tf.concat(features, axis=0)
                knn_indices = tf.concat(knn_indices, axis=0)
                knn_scale = tf.concat(knn_scale, axis=0)
                triangles = tf.concat(triangles, axis=0)
                nv = tf.stack(nv, axis=0)
                mf = tf.stack(mf, axis=0)

                points = points[:,:3]
                yield points, features, knn_indices, knn_scale, triangles, nv, mf

                new_offset_idx = 0
                points, features, knn_indices, knn_scale, triangles, nv, mf = [], [], [], [], [], [], []

        if (not drop_remainder) and (len(points)>0):
            points = tf.concat(points, axis=0)
            features = tf.concat(features, axis=0)
            knn_indices = tf.concat(knn_indices, axis=0)
            knn_scale = tf.concat(knn_scale, axis=0)
            triangles = tf.concat(triangles, axis=0)
            nv = tf.stack(nv, axis=0)
            mf = tf.stack(mf, axis=0)

            points = points[:,:3]
            yield points, features, knn_indices, knn_scale, triangles, nv, mf


