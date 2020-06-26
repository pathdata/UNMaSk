import os
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
import pickle

from sccnn_classifier import sccnn_classifier


def run(opts_in):
    if os.path.isdir(os.path.join(opts_in.exp_dir, 'code')):
        rmtree(os.path.join(opts_in.exp_dir, 'code'))
        os.makedirs(os.path.join(opts_in.exp_dir, 'code'))

    if not os.path.isdir(opts_in.exp_dir):
        os.makedirs(opts_in.exp_dir)
        os.makedirs(opts_in.checkpoint_dir)
        os.makedirs(opts_in.log_train_dir)
        os.makedirs(os.path.join(opts_in.exp_dir, 'code'))

    network = sccnn_classifier.SccnnClassifier(batch_size=opts_in.batch_size,
                                               image_height=opts_in.image_height,
                                               image_width=opts_in.image_width,
                                               in_feat_dim=opts_in.in_feat_dim,
                                               in_label_dim=opts_in.in_label_dim,
                                               num_of_classes=opts_in.num_of_classes,
                                               tf_device=opts_in.tf_device)

    curr_file_path = os.path.realpath(__file__)
    curr_file_dir = os.path.dirname(curr_file_path)
    copyfile(os.path.join(curr_file_dir, '..', 'train_network_main.py'),
             os.path.join(opts_in.exp_dir, 'code', 'train_network_main.py'))
    copytree(os.path.join(curr_file_dir, '..', 'sccnn_classifier'),
             os.path.join(opts_in.exp_dir, 'code', 'sccnn_classifier'))

    pickle.dump(opts_in, open(os.path.join(opts_in.exp_dir, 'code', 'opts.p'), 'wb'))

    network = network.run_training(opts=opts_in)

    return network
