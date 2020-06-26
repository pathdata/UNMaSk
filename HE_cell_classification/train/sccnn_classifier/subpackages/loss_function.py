import tensorflow as tf
import numpy as np


def loss_function(network, logits=None, labels=None, weighted_loss_per_class=None):

    if weighted_loss_per_class is None:
        weighted_loss_per_class = np.ones(network.num_of_classes)

    if labels is None:
        labels = network.labels

    if logits is None:
        logits = network.logits

    # labels = tf.cast(labels, tf.int64)
    # labels = tf.expand_dims(labels, dim=1)
    # labels = tf.one_hot(labels, depth=network.num_of_classes, axis=3)

    epsilon = 1e-6
    clipped_logits = tf.clip_by_value(logits, epsilon, 1.0 - epsilon)
    log_loss = -labels * tf.log(clipped_logits) - (1.0 - labels) * tf.log(1.0 - clipped_logits)
    weighted_log_loss = tf.multiply(log_loss, weighted_loss_per_class)
    cross_entropy_log_loss = tf.reduce_sum(weighted_log_loss)
    loss = cross_entropy_log_loss
    _ = tf.summary.scalar('Loss', loss)

    return loss
