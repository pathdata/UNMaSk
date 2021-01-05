import tensorflow as tf
import glob
import os
#import matlab.engine
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import cv2
import 

from dcis_segmentation.subpackages import Patches


def make_sub_dirs(opts, sub_dir_name):

    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'mask_image', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mask_image', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))



def calculate_magnitude(od):
    channel = cv2.split(od)
    square_b = cv2.pow(channel[0], 2)
    square_g = cv2.pow(channel[1], 2)
    square_r = cv2.pow(channel[2], 2)
    square_bgr = square_b + square_g + square_r
    magnitude = cv2.sqrt(square_bgr)

    return magnitude;


def normaliseOD(od, magnitude):
    channels = cv2.split(od)  # , (channels[0],channels[1],channels[2]))

    od_norm_b = cv2.divide(channels[0], magnitude);
    od_norm_g = cv2.divide(channels[1], magnitude);
    od_norm_r = cv2.divide(channels[2], magnitude);

    od_norm = cv2.merge((od_norm_b, od_norm_g, od_norm_r))

    return od_norm;

def clean_artifact(cws_img, image_path_full):

    I = cws_img.transpose()
    k, width, height = I.shape
    I = I.reshape(k, width * height);
    I = np.float32(I)

    od = cv2.max(I, 1)

    grey_angle = 0.2;

    magnitude_threshold = 0.05;

    channels = cv2.split(od);
    #
    magnitude = np.zeros(od.shape)
    #
    background = 245
    #
    channels[0] /= background

    od = cv2.merge(channels)

    od = cv2.log(od)

    od *= (1 / cv2.log(10)[0])

    od = -od
    od = od.reshape(3, width, height).transpose()
    magnitude = calculate_magnitude(od)

    tissue_and_artefact_mask = (magnitude > magnitude_threshold);

    od_norm = normaliseOD(od, magnitude);

    chan = cv2.split(od_norm)

    grey_mask = (chan[0] + chan[1] + chan[2]) >= (.cos(grey_angle) * cv2.sqrt(3)[0])

    other_colour_mask = (chan[2] > chan[1]) | (chan[0] > chan[1])

    mask = grey_mask | other_colour_mask

    mask = (255 - mask) & tissue_and_artefact_mask
    mask1 = mask.astype(np.int8)

    clean = cv2.bitwise_and(cws_img, cws_img, mask=mask1)
    clean = cv2.bitwise_not(clean)

    clean = cv2.bitwise_and(clean, clean, mask=mask1)
    clean = cv2.bitwise_not(clean)

    write_mask1 = mask.astype(np.uint8)*255

    
    return (clean,write_mask1)



def generate_network_output(opts, sub_dir_name, network, sess, logits):

    make_sub_dirs(opts, sub_dir_name)
    if opts.tissue_segment_dir == '':
        files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
    else:
        files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat')))
    for i in range(len(files_tissue)):
        file_name = os.path.basename(files_tissue[i])
        file_name = file_name[:-4]
        if not os.path.isfile(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat')):
            print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            if opts.pre_process:
                workspace = sio.loadmat(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name,
                                                     file_name + '.mat'))
                matlab_output = workspace['matlab_output']
                feat = np.array(matlab_output['feat'][0][0])
            else:
                feat = image_path_full

            patch_obj = Patches.Patches(
                img_patch_h=opts.image_height, img_patch_w=opts.image_width,
                stride_h=opts.stride_h, stride_w=opts.stride_w,
                label_patch_h=opts.label_height, label_patch_w=opts.label_width)

            image_patches = patch_obj.extract_patches(feat)
            opts.num_examples_per_epoch_for_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_for_train, opts.label_height,
                                      opts.label_width, opts.num_of_classes], dtype=np.float32)
            train_count = int((opts.num_examples_per_epoch_for_train / opts.batch_size) + 1)

            start = 0
            start_time = time.time()
            for step in range(train_count):
                end = start + opts.batch_size
                data_train = image_patches[start:end, :, :, :]
                data_train = data_train.astype(np.float32, copy=False)
                data_train_float32 = data_train / 255.0
                logits_out = sess.run(
                    logits,
                    feed_dict={network.images: data_train_float32,
                               })
                label_patches[start:end] = logits_out

                if end + opts.batch_size > opts.num_examples_per_epoch_for_train - 1:
                    end = opts.num_examples_per_epoch_for_train - opts.batch_size

                start = end

            output = patch_obj.merge_patches(label_patches)
            mat = {'output': output}
            mat_file_name = file_name + '.mat'
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name), mat)

            mat_output = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name))
            DCIS_prob = mat_output['output'][:, :, 1] > 0.2

            DCIS_mask = DCIS_prob.astype(np.uint8)*255
            #cv2.imwrite(os.path.join(opts.results_dir, 'mask_image', sub_dir_name, file_name + '.png'), DCIS_mask)

            cws_img = cv2.imread(image_path_full)
            # img = cv2.imread(os.path.join(input_dir, im))
            img, mask = clean_artifact(cws_img, image_path_full)
            #cv2.imwrite(os.path.join(opts.results_dir, 'mask_image', sub_dir_name, file_name + '_post.png'), img)


            post_img_mask = cv2.bitwise_and(mask,DCIS_mask)
            cv2.imwrite(os.path.join(opts.results_dir, 'mask_image', sub_dir_name, file_name + '.png'), post_img_mask)

            duration = time.time() - start_time
            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            print('Already segmented %s/%s\n' % (sub_dir_name, file_name), flush=True)


def generate_output(network, opts, save_pre_process=True, network_output=True, post_process=True):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    # eng = matlab.engine.start_matlab()
    # eng.addpath('dcis_segmentation')
    # eng.eval('run initialize_matlab_variables.m', nargout=0)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)

        for cws_n in range(0, len(cws_sub_dir)):
            curr_cws_sub_dir = cws_sub_dir[cws_n]
            print(curr_cws_sub_dir)
            sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
            # if save_pre_process:
            #     pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

            if network_output:
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network,
                                        sess=sess, logits=logits)

            # if post_process:
            #     post_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path)
        print('---------------------------------------------------------------\n', flush=True)
        print('---------------------------------------------------------------\n', flush=True)
        print('---------------------------------------------------------------\n', flush=True)

        # eng = matlab.engine.start_matlab()
        # eng.addpath('dcis_segmentation')
        # eng.eval('run initialize_matlab_variables.m', nargout=0)
        # if save_pre_process:
        #     pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

        if network_output:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network,
                                    sess=sess, logits=logits)

        # if post_process:
        #     post_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

    return opts.results_dir
