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
    labels = (labels - 1).astype('int64')
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'in_feat/shape': int64_list_feature(in_feat.shape),
                'in_feat/data': bytes_feature(in_feat.tostring()),
                'labels/data': int64_feature(labels)
            }))

    return tf_example.SerializeToString()


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'in_feat/shape': tf.FixedLenFeature([3], tf.int64),
            'in_feat/data': tf.FixedLenFeature([], tf.string),
            'labels/data': tf.FixedLenFeature([1], tf.int64)
        })
    in_feat = tf.decode_raw(features['in_feat/data'], tf.uint8)
    in_feat = tf.reshape(in_feat, tf.cast(features['in_feat/shape'], tf.int32))
    in_feat = tf.divide(tf.cast(in_feat, tf.float32), 255)

    labels_data = tf.cast(features['labels/data'], tf.uint8)

    return in_feat, labels_data


def augment(data_in):
    data_out = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), data_in)
    data_out = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), data_out)
    return data_out


def random_variations(data_in, in_feat_dim=3, num_feat_aug=3):
    if in_feat_dim > num_feat_aug:
        data_ = data_in[:, :, :, 0:num_feat_aug]
        data2_ = data_in[:, :, :, num_feat_aug:data_in.shape[3]]
    else:
        data_ = data_in
        data2_ = None

    randomly_adjust_data = tf.map_fn(
        lambda img: tf.image.random_brightness(img, max_delta=0.5), data_)
    randomly_adjust_data = tf.map_fn(
        lambda img: tf.image.random_contrast(img, lower=0.8, upper=1), randomly_adjust_data)
    randomly_adjust_data = tf.map_fn(
        lambda img: tf.image.random_hue(img, max_delta=0.01), randomly_adjust_data)
    randomly_adjust_data = tf.map_fn(
        lambda img: tf.image.random_saturation(img, lower=0.8, upper=1), randomly_adjust_data)

    if in_feat_dim > num_feat_aug:
        data_out = tf.concat(values=[randomly_adjust_data, data2_], axis=3)
    else:
        data_out = randomly_adjust_data

    return data_out


def get_data_set(filename, num_epochs, shuffle_size, batch_size, prefetch_buffer):
    data_set = tf.data.TFRecordDataset(filename)
    data_set = data_set.map(decode)
    data_set = data_set.prefetch(prefetch_buffer)
    data_set = data_set.repeat(num_epochs)
    data_set = data_set.shuffle(shuffle_size)
    data_set = data_set.batch(batch_size)
    return data_set
