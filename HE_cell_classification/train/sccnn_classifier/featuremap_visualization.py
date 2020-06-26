import os
import glob
import numpy as np
import math
import tensorflow as tf
import pickle
import sys

from tensorflow.contrib.tensorboard.plugins import projector
from sccnn_classifier import sccnn_classifier as sccnn
from sccnn_classifier.subpackages import Patches as Patches
from sccnn_classifier.subpackages import create_sprite_image, create_tsv
from sccnn_classifier import subpackages


def run(opts_in):
    exp_dir = opts_in['exp_dir']
    data_dir = opts_in['data_dir']
    cws_dir = opts_in['cws_dir']
    # noinspection PyBroadException
    try:
        opts = pickle.load(open(os.path.join(exp_dir, 'code', 'opts.p'), 'rb'))
    except Exception:
        sys.modules['subpackages'] = subpackages
        opts = pickle.load(open(os.path.join(exp_dir, 'code', 'opts.p'), 'rb'))

    opts.exp_dir = exp_dir
    opts.data_dir = data_dir
    opts.cws_dir = cws_dir
    opts.log_train_dir = os.path.join(opts.exp_dir, 'logs', 'embedding')
    opts.batch_size = 100

    if not os.path.isdir(opts.log_train_dir):
        os.makedirs(opts.log_train_dir)

    max_size_sprite_image = [8192, 8192]  # TensorFlow doesn't support beyond this size
    size_of_patch = [20, 20]
    patch_obj = Patches.Patches(patch_h=opts.image_height, patch_w=opts.image_width, num_examples_per_patch=1)

    files = glob.glob(os.path.join(opts.data_dir, 'Da*.csv'))
    name_to_visualise_variable = "Convolution_6"

    to_embed_count = max_size_sprite_image[0] * max_size_sprite_image[1]
    max_num_images = np.divide(max_size_sprite_image, size_of_patch)
    max_num_images = max([math.floor(float(x)) for x in max_num_images])
    max_num_images = max_num_images ** 2

    if max_num_images > to_embed_count:
        max_num_images = to_embed_count

    images_all = []
    labels_all = []
    total = 0
    for i in range(len(files)):
        full_file_name = os.path.basename(files[i])
        file_name, _ = os.path.splitext(full_file_name)
        image_patches, labels, cell_id = patch_obj.extract_patches(
            input_image=os.path.join(opts.cws_dir, file_name + '.jpg'),
            input_csv=os.path.join(opts.data_dir, full_file_name))

        for j in range(image_patches.shape[0]):
            labels_all.append(labels[j])
            images_all.append(image_patches[j])

        total += image_patches.shape[0]
        if total > int(max_num_images):
            break

    labels_all = labels_all[0:max_num_images]
    images_all = images_all[0:max_num_images]

    images_all = np.array(images_all)
    labels_all = np.array(labels_all)

    create_sprite_image.create_sprite_image(images_all, size_of_patch, opts.log_train_dir)
    create_tsv.create_tsv(labels_all, opts.log_train_dir)

    network = sccnn.SccnnClassifier(batch_size=opts.batch_size,
                                    image_height=opts.image_height,
                                    image_width=opts.image_width,
                                    in_feat_dim=opts.in_feat_dim,
                                    in_label_dim=opts.in_label_dim,
                                    num_of_classes=opts.num_of_classes)

    # batch_train = np.zeros([opts_in.batch_size, opts_in.image_height, opts_in.image_width, opts_in.in_feat_dim],
    #                        dtype=np.float32)
    _, output = network.inference(is_training=False)
    convolution_5 = output['convolution_5']

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path)
        else:
            sess.run(init)
            print('No checkpoint file found')

        opts.num_examples_per_epoch_for_train = images_all.shape[0]
        label_patches = np.zeros(
            [opts.num_examples_per_epoch_for_train, convolution_5.shape[1], convolution_5.shape[2],
             convolution_5.shape[3]],
            dtype=np.float32)

        train_count = int((opts.num_examples_per_epoch_for_train / opts.batch_size) + 1)

        start = 0

        for step in range(train_count):
            end = start + opts.batch_size
            data_train = images_all[start:end, :, :, :]
            data_train = data_train.astype(np.float32, copy=False)
            data_train_float32 = data_train / 255.0
            out = sess.run(
                convolution_5,
                feed_dict={network.images: data_train_float32,
                           })
            label_patches[start:end] = out

            if end + opts.batch_size > opts.num_examples_per_epoch_for_train - 1:
                end = opts.num_examples_per_epoch_for_train - opts.batch_size

            start = end

    label_patches = np.reshape(label_patches,
                               newshape=[opts.num_examples_per_epoch_for_train,
                                         label_patches.shape[1] * label_patches.shape[2] * label_patches.shape[3]])
    tf.reset_default_graph()
    embedding_var = tf.Variable(label_patches, trainable=False, name=name_to_visualise_variable)

    print(embedding_var.shape)

    summary_writer = tf.summary.FileWriter(opts.log_train_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = os.path.join(opts.log_train_dir, 'metadata.tsv')  # 'metadata.tsv'

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = os.path.join(opts.log_train_dir, 'sprite_image.png')  # 'mnist_digits.png'
    embedding.sprite.single_image_dim.extend(size_of_patch)

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(opts.log_train_dir, "model.ckpt"), 1)
