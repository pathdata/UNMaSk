import tensorflow as tf
import scipy.io as sio
import os
import random
from dcis_segmentation.subpackages import Patches, data_utils, tools


def read_mat_file(file_name):
    workspace = sio.loadmat(file_name)
    data = workspace['data']
    labels = workspace['labels']
    return data, labels


def write_to_tf(files, save_path, save_filename):
    random.shuffle(files)
    patch_obj = Patches.Patches(
        img_patch_h=600, img_patch_w=600,
        stride_h=400, stride_w=400,
        label_patch_h=600, label_patch_w=600)
    print('Writing', os.path.join(save_path, save_filename + '.tfrecords'))
    tf_writer = tf.python_io.TFRecordWriter(os.path.join(save_path, save_filename + '.tfrecords'))
    num_examples = 0
    tools.printProgressBar(0, len(files), prefix='Progress:', suffix='Complete', length=50)

    for file_n in range(0, len(files)):
        curr_train_file = str(files[file_n])
        # print('Processing ' + curr_train_file)
        tools.printProgressBar(file_n + 1, len(files), prefix='Progress:', suffix='Complete', length=50)
        data, labels = read_mat_file(file_name=curr_train_file)
        data = patch_obj.extract_patches(data)
        labels = patch_obj.extract_patches(labels)
        for i in range(data.shape[0]):
            tf_serialized_example = data_utils.encode(in_feat=data[i], labels=labels[i])
            tf_writer.write(tf_serialized_example)
            num_examples += 1

    out_dict = {'num_examples': num_examples}
    sio.savemat(os.path.join(save_path, save_filename + '.mat'), out_dict)
    tf_writer.close()


def run(opts_in):
    save_path = opts_in['save_path']
    main_file_path = opts_in['main_file_path']
    train_filename = opts_in['train_filename']
    valid_filename = opts_in['valid_filename']

    train_files = list(main_file_path.joinpath('mat').glob('Train*.mat'))
    valid_files = list(main_file_path.joinpath('mat').glob('Valid*.mat'))

    write_to_tf(files=train_files, save_path=save_path, save_filename=train_filename)
    write_to_tf(files=valid_files, save_path=save_path, save_filename=valid_filename)
