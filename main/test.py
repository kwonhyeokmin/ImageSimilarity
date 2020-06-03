import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os

# change tensorflow level for removing warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test1():
    assert 1 == 0


def test2():
    a = tf.constant(np.array([1, 0, 0, 1]), dtype=tf.float32)

    result = tf.constant(np.array([3, 0, 0, 3]), dtype=tf.float32)
    assert tf.reduce_all(tf.equal(3 * a, result))


def test3():
    a = tf.constant(np.array([1, 2, 3, 4]), dtype=tf.float32)
    state_diag = tf.linalg.diag(a)

    result = tf.constant(np.array([[1, 0, 0, 0],
                                   [0, 2, 0, 0],
                                   [0, 0, 3, 0],
                                   [0, 0, 0, 4]]), dtype=tf.float32)
    assert tf.reduce_all(tf.equal(state_diag, result))


def test4():
    a = tf.constant(np.array([1, 2, 3, 4]), dtype=tf.float32)
    b = tf.constant(np.array([3, 1, 3, 1]), dtype=tf.float32)
    result = tf.maximum(0, a - b)
    answer = tf.constant(np.array([0, 1, 0, 3]), dtype=tf.float32)
    assert tf.reduce_all(tf.equal(answer, result))