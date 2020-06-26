import tensorflow as tf
import scipy.io as sio
import os
import random
from sccnn_classifier.subpackages import tools
from sccnn_classifier.subpackages import data_utils


def read_mat_file(file_name):
    workspace = sio.loadmat(file_name)
    data = workspace['data']
    labels = workspace['labels']
    return data, labels

def run(main_input_path, save_tf_path, train_tf_filename, valid_tf_filename):

    train_main_file_path = os.path.join(main_input_path,'training')
    valid_main_file_path = os.path.join(main_input_path,'validation')

    train_files = list(train_main_file_path.glob('*.mat'))
    valid_files = list(valid_main_file_path.glob('*.mat'))

    write_to_tf(files=train_files, save_path=save_tf_path, save_filename=train_tf_filename)
    write_to_tf(files=valid_files, save_path=save_tf_path, save_filename=valid_tf_filename)


def write_to_tf(files, save_path, save_filename):
    random.shuffle(files)
    print('Writing', os.path.join(save_path, save_filename + '.tfrecords'))
    tf_writer = tf.python_io.TFRecordWriter(os.path.join(save_path, save_filename + '.tfrecords'))
    num_examples = 0
    tools.printProgressBar(0, len(files), prefix='Progress:', suffix='Complete', length=50)

    for file_n in range(0, len(files)):
    # for file_n in range(0, 10):
        curr_train_file = str(files[file_n])
        tools.printProgressBar(file_n + 1, len(files), prefix='Progress:', suffix='Complete', length=50)
        data, labels = read_mat_file(file_name=curr_train_file)
        tf_serialized_example = data_utils.encode(in_feat=data, labels=labels)
        tf_writer.write(tf_serialized_example)
        num_examples += 1

    out_dict = {'num_examples': num_examples}
    sio.savemat(os.path.join(save_path, save_filename + '.mat'), out_dict)
    tf_writer.close()




