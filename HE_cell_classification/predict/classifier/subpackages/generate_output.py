import tensorflow as tf
import glob
import os
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import pandas as pd
from scipy import stats
import cv2

from classifier.subpackages import Patches, h5


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'csv', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'csv', sub_dir_name))
    # if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name)):
    #     os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name))




color_coding = {'f': (0, 255, 255),
                'l': (255, 0, 0),
                't': (0, 255, 0),
                'o': (255, 255, 255)
                }

def put_markers_cv(X,Y, im, cell_class_names):
    r = 5
    Y = list(Y)  # row
    X = list(X)  # col
    for i in range(len(X)):
        cv2.circle(im, (X[i], Y[i]), r, color=color_coding[cell_class_names[i]], thickness=-1)
    return im



def get_n_mode(arr, n = 9):
    assert len(arr)%n == 0
    arr = arr.reshape(-1, n)
    m = stats.mode(arr, axis = 1)[0]
    return np.array(m)


def generate_network_output(opts, sub_dir_name, network, sess, logits_labels, csv_detection_results_dir):
    make_sub_dirs(opts, sub_dir_name)
    if opts.tissue_segment_dir == '':
        files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
    else:
        files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat',sub_dir_name, 'Da*.mat')))

    for i in range(len(files_tissue)):
        file_name = os.path.basename(files_tissue[i])
        file_name = file_name[:-4]
        if not os.path.isfile(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat')):
            print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            # if opts.pre_process:
            #     feat = h5.h5read(
            #         filename=os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, file_name + '.h5'),
            #         data_name='feat')
            # else:
            feat = image_path_full

            patch_obj = Patches.Patches(patch_h=opts.image_height, patch_w=opts.image_width)

            image_patches, labels, cell_id = patch_obj.extract_patches(
                input_image=feat,
                input_csv=os.path.join(csv_detection_results_dir, file_name + '.csv'))
            opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_train, opts.in_label_dim], dtype=np.float32)
            train_count = int((opts.num_examples_per_epoch_train / opts.batch_size) + 1)

            start = 0
            start_time = time.time()

            if image_patches.shape[0] != 0 and opts.batch_size > opts.num_examples_per_epoch_train:
                image_patches_temp = image_patches
                for rs_var in range(int((opts.batch_size / opts.num_examples_per_epoch_train))):
                    image_patches_temp = np.concatenate((image_patches_temp, image_patches), axis=0)

                image_patches = image_patches_temp

            opts.num_examples_per_epoch_train_temp = image_patches.shape[0]

            if image_patches.shape[0] != 0:
                label_patches = np.zeros([opts.num_examples_per_epoch_train_temp, opts.in_label_dim], dtype=np.float32)
                for step in range(train_count):
                    end = start + opts.batch_size
                    data_train = image_patches[start:end, :, :, :]
                    data_train = data_train.astype(np.float32, copy=False)
                    data_train_float32 = data_train / 255.0
                    logits_out = sess.run(
                        logits_labels,
                        feed_dict={network.images: data_train_float32,
                                   })
                    label_patches[start:end] = np.squeeze(logits_out, axis=1) + 1

                    if end + opts.batch_size > opts.num_examples_per_epoch_train_temp - 1:
                        end = opts.num_examples_per_epoch_train_temp - opts.batch_size

                    start = end

                label_patches = label_patches[0:opts.num_examples_per_epoch_train]
            duration = time.time() - start_time
            mat = {'output': label_patches,
                   'labels': labels,
                   'cell_ids': cell_id}
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat'), mat)

            class_labels = set(mat['cell_ids'])

            mat_output_val = mat['output'][0:label_patches.shape[0]]

            result = get_n_mode(mat_output_val)
            mat = {'output': label_patches,
                   'labels': labels,
                   'cell_ids': cell_id,
                   'class': result}
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat'), mat)


            class_scnn = np.array(result)#np.array(workspace['mat'][0]['class'][0])
            #print(len(class_scnn))

            df = pd.read_csv(os.path.join(csv_detection_results_dir, file_name + '.csv'))

            X = list(df['V2'])
            Y = list(df['V3'])
            #
            dfD = pd.DataFrame(columns=['V1', 'V2', 'V3'])
            #
            class_V1=[]
            for i in range(len(class_scnn)):

                if class_scnn[i] == 1:
                    class_V1.append('f')
                elif class_scnn[i] == 2:
                    class_V1.append('l')
                elif class_scnn[i] == 3:
                    class_V1.append('t')
                elif class_scnn[i] == 4:
                    class_V1.append('o')
                else:
                    continue

                dfD = dfD.append({'V1': class_V1[i], 'V2': X[i], 'V3': Y[i]}, ignore_index=True)

            dfD.to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, file_name + '.csv'),index=False)

            img = cv2.imread(feat)

            im_out = put_markers_cv(list(dfD['V2']), list(dfD['V3']), img, list(dfD['V1']))

            cv2.imwrite(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, file_name + '.jpg'), im_out)

            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            print('Already classified %s/%s\n' % (sub_dir_name, file_name), flush=True)



def generate_output(network, opts, save_pre_process=True, network_output=True):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    print(cws_sub_dir)
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)



    for cws_n in range(0, len(cws_sub_dir)):
        curr_cws_sub_dir = cws_sub_dir[cws_n]
        print(curr_cws_sub_dir, flush=True)
        sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
        csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)
        print(csv_detection_results_dir)

        opts.checkpoint_dir = opts.exp_dir




        if network_output:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
                assert ckpt, "No Checkpoint file found"
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                        logits_labels=logits_labels,
                                        csv_detection_results_dir=csv_detection_results_dir)



    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, network_output=True):
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)



        if network_output:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                    logits_labels=logits_labels,
                                    csv_detection_results_dir=csv_detection_results_dir)



    return opts.results_dir