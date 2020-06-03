import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
import numpy as np


class BaseNet(Model):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dropout1 = layers.Dropout(0.1)
        self.dense2 = layers.Dense(128, activation='relu', name='dense_2')
        self.dropout2 = layers.Dropout(0.1)
        self.dense3 = layers.Dense(128, activation='relu', name='dense_3')
        self.dropout3 = layers.Dropout(0.1)
        self.dense4 = layers.Dense(128, activation='relu', name='dense_4')

    def call(self, inputs, **kwargs):
        net = self.flatten(inputs)
        net = self.dense1(net)
        net = self.dropout1(net)
        net = self.dense2(net)
        net = self.dropout2(net)
        net = self.dense3(net)
        net = self.dropout3(net)
        output = self.dense4(net)
        return output


def create_model():
    net = BaseNet()
    net(tf.random.normal((1, 28, 28)))
    return net
