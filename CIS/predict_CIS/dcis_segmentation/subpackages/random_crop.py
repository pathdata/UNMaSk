import tensorflow as tf


def random_crop(opts, images, labels):
    concat = tf.concat(values=[images, labels], axis=3)
    cropped = tf.map_fn(lambda img: tf.random_crop(img, [opts.crop_height, opts.crop_width,
                                                         opts.in_feat_dim + opts.in_label_dim]), concat)
    images = cropped[:, :, :, 0:opts.in_feat_dim]
    labels = cropped[:, :, :, opts.in_feat_dim:opts.in_feat_dim + opts.in_label_dim]

    return images, labels
