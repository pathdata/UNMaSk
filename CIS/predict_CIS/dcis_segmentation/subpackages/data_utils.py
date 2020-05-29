import tensorflow as tf
import numpy as np


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encode(in_feat, labels):
    if len(labels.shape) == 2:
        labels = np.expand_dims(labels, axis=2)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'in_feat/shape': int64_list_feature(in_feat.shape),
                'in_feat/data': bytes_feature(in_feat.tostring()),
                'labels/shape': int64_list_feature(labels.shape),
                'labels/data': bytes_feature(labels.tostring())}))

    return tf_example.SerializeToString()


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'in_feat/shape': tf.FixedLenFeature([3], tf.int64),
            'in_feat/data': tf.FixedLenFeature([], tf.string),
            'labels/shape': tf.FixedLenFeature([3], tf.int64),
            'labels/data': tf.FixedLenFeature([], tf.string)
        })
    in_feat = tf.decode_raw(features['in_feat/data'], tf.uint16)
    in_feat = tf.reshape(in_feat, tf.cast(features['in_feat/shape'], tf.int32))
    in_feat = tf.divide(tf.cast(in_feat, tf.float32), 65535)

    labels_data = tf.decode_raw(features['labels/data'], tf.uint8)
    labels_data = tf.reshape(labels_data, tf.cast(features['labels/shape'], tf.int32))
    labels_data = tf.cast(labels_data, tf.float32)
    labels0 = tf.subtract(labels_data[:, :, 0:1], 1.0)
    labels1 = tf.subtract(2.0, labels_data[:, :, 0:1])
    labels2 = labels_data[:, :, 1:2]
    labels_data = tf.concat(values=[labels1, labels0, labels2, labels2], axis=2)

    return in_feat, labels_data


def augment(data_in, labels_in):
    data_concat = tf.concat(values=[data_in, labels_in], axis=3)
    data_concat = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), data_concat)
    data_concat = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), data_concat)
    data_out = data_concat[:, :, :, 0:3]
    labels_out = data_concat[:, :, :, 3:7]
    return data_out, labels_out


def random_variations(data_in):
    randomly_adjust_data = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=0.5), data_in)
    randomly_adjust_data = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.8, upper=1), randomly_adjust_data)
    randomly_adjust_data = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.01), randomly_adjust_data)
    randomly_adjust_data = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0.8, upper=1), randomly_adjust_data)

    return randomly_adjust_data


def get_data_set(filename, num_epochs, shuffle_size, batch_size, prefetch_buffer):
    data_set = tf.data.TFRecordDataset(filename)
    data_set = data_set.map(decode)
    data_set = data_set.prefetch(prefetch_buffer)
    data_set = data_set.repeat(num_epochs)
    data_set = data_set.shuffle(shuffle_size)
    data_set = data_set.batch(batch_size)
    return data_set
