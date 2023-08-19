import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


class CircNet(tf.Module):

    def __init__(self, dim, T, S=2, knn=50, multiplier=16):
        super(CircNet, self).__init__()
        self.K = knn
        self.beta = multiplier
        self.slice = T*S*dim
        self.new_shape = [-1, T, S, dim]

        # settings for graph weights and channels
        self.gw_shape = [-1,knn,32,multiplier]  
        self.gw_channels = 32*multiplier

        self.layer_0 = tf.keras.layers.Dense(32, activation='relu') 
        self.layer_1 = tf.keras.layers.Dense(32, activation='relu')
        self.layer_2 = tf.keras.layers.Dense(32, activation='relu')

        # compute the graph convolution weights
        self.graph_weights = tf.keras.layers.Dense(self.gw_channels, activation='relu')
        self.layer_3 = tf.keras.layers.Dense(1024, activation='relu')
        self.layer_7 = tf.keras.layers.Dense(1024, activation='relu')
        self.predict = tf.keras.layers.Dense(T*S*dim+T, activation=None)


    def __call__(self, inputs, training=None):

        out = self.layer_0(inputs, training=training)
        out = self.layer_1(out, training=training)
        out = self.layer_2(out, training=training)

        # graph convolution to get global feature of the patch
        Graph_W = self.graph_weights(out, training=training)
        Hada_out = tf.expand_dims(out, axis=3)*tf.reshape(Graph_W, shape=self.gw_shape)
        out = tf.reshape(tf.reduce_sum(Hada_out, axis=1), [-1, self.gw_channels])

        out = self.layer_3(out, training=training)
        out = self.layer_7(out, training=training)
        out = self.predict(out, training=training)

        out =tf.squeeze(out)
        offsets = tf.reshape(out[...,:self.slice], self.new_shape)
        logits = tf.reshape(out[...,self.slice:], self.new_shape[:-2]+[1])
        logits = tf.squeeze(logits)

        return offsets, logits
