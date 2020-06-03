from tensorflow.keras.datasets import mnist
import tensorflow as tf
from functools import partial
import itertools
import numpy as np
import time
from model import create_model
import os
import os.path as osp
from pathlib import Path
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def generate(subset='train'):
    number_of_dataset = 1000
    if subset == 'train':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.array(x_train) / .255
        x_train = x_train[:number_of_dataset]
        train_dataset = [[x, y] for x, y in zip(x_train, y_train)]
        train_dataset = [[x[0][0], x[1][0], 0 if x[0][1] == x[1][1] else 1] for x in
                         itertools.permutations(train_dataset, 2)]

        for img_f, img_s, state in train_dataset:
            img_f = tf.constant(img_f, dtype=tf.float32)
            img_s = tf.constant(img_s, dtype=tf.float32)
            state = tf.constant(state, dtype=tf.float32)

            yield img_f, img_s, state


def loss(vector_f, vector_s, state):
    margin = 1000
    distance = tf.square(vector_f - vector_s)
    state_diag = tf.linalg.diag(state)
    # 같은 이미지: distance, 다른 이미지: 1 - distance
    result = tf.matmul(state_diag, margin - 2*distance) + distance
    return tf.reduce_mean(tf.maximum(0.0, result))


@tf.function
def train_step(img_f, img_s, state, model, optimizer):
    with tf.GradientTape() as tape:
        vector_f = model(img_f)
        vector_s = model(img_s)

        loss_value = loss(vector_f, vector_s, state)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value


if __name__ == '__main__':
    # define directory
    ROOT_DIR = Path.cwd().parent
    OUTPUT_DIR = osp.join(ROOT_DIR, 'output')
    MODEL_DIR = osp.join(OUTPUT_DIR, 'model_dump')

    # model config
    batch_size = 32
    num_epoch = 10

    # load model
    logger.info('Create model start')
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    logger.info('Create model success')

    # make dataset for train
    logger.info('Make dataset with mnist')
    train_gen = partial(generate, subset='train')
    train_dataset = tf.data.Dataset.from_generator(
        train_gen, (tf.float32, tf.float32, tf.float32)
    )
    batch_generator = train_dataset.shuffle(40).batch(batch_size)
    logger.info('Make dataset success')

    logger.info('Training start')
    start_time = time.time()
    for epoch in range(num_epoch):
        for i, (img_f, img_s, state) in enumerate(batch_generator):
            loss_value = train_step(img_f, img_s, state, model, optimizer)
        logger.info('Epoch: {} Time: {:.2}s | Loss: {:.8f}'.format(epoch + 1, time.time() - start_time, loss_value))
        start_time = time.time()
        model.save_weights(
            os.path.join(MODEL_DIR, 'model_epoch_{}.h5'.format(epoch + 1)))
