import tensorflow as tf


def loss_function(network, weighted_loss_per_class):

    labels = tf.cast(network.labels - 1, tf.int64)
    labels = tf.expand_dims(labels, dim=1)
    labels = tf.one_hot(labels, depth=network.num_of_classes, on_value=1.0, off_value=0.0, axis=3)

    epsilon = 1e-6
    clipped_logits = tf.clip_by_value(network.logits, epsilon, 1.0 - epsilon)
    log_loss = -labels * tf.log(clipped_logits) - (1.0 - labels) * tf.log(1.0 - clipped_logits)
    weighted_log_loss = tf.multiply(log_loss, weighted_loss_per_class)
    cross_entropy_log_loss = tf.reduce_sum(weighted_log_loss)
    network.loss = cross_entropy_log_loss
    _ = tf.summary.scalar('Loss', network.loss)

    return network.loss
