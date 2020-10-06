import os
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
import scipy.io as sio
import pickle

from classifier.sccnn_classifier import SccnnClassifier
from classifier.subpackages import NetworkOptions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = NetworkOptions.NetworkOptions(exp_dir=os.path.normpath(os.path.join(os.getcwd(), 'ExpDir-POET-S')),
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=51,
                                     image_width=51,
                                     in_feat_dim=3,
                                     in_label_dim=1,
                                     num_of_classes=4,
                                     batch_size=1000,
                                     num_of_epoch=500,
                                     data_dir=os.path.normpath('Data_IHC_CV1/'),
                                     train_data_filename='TrainData191026_POET.h5',
                                     valid_data_filename='ValidData191026_POET.h5',
                                     current_epoch_num=0)

if os.path.isdir(os.path.join(opts.exp_dir, 'code')):
    rmtree(os.path.join(opts.exp_dir, 'code'))
    os.makedirs(os.path.join(opts.exp_dir, 'code'))

if not os.path.isdir(opts.exp_dir):
    os.makedirs(opts.exp_dir)
    os.makedirs(opts.checkpoint_dir)
    os.makedirs(opts.log_train_dir)
    os.makedirs(os.path.join(opts.exp_dir, 'code'))


Network = SccnnClassifier(batch_size=opts.batch_size,
                                           image_height=opts.image_height,
                                           image_width=opts.image_width,
                                           in_feat_dim=opts.in_feat_dim,
                                           in_label_dim=opts.in_label_dim,
                                           num_of_classes=opts.num_of_classes,
                                           tf_device=opts.tf_device)

copyfile('Train_Network_Main.py', os.path.join(opts.exp_dir, 'code', 'Train_Network_Main.py'))
copyfile('Generate_Output_TA.py', os.path.join(opts.exp_dir, 'code', 'Generate_Output.py'))
copytree('classifier', os.path.join(opts.exp_dir, 'code', 'classifier'))


mat = {'opts': opts}
sio.savemat(os.path.join(opts.exp_dir, 'code', 'opts.mat'), mat)
pickle.dump(opts, open(os.path.join(opts.exp_dir, 'code', 'opts.p'), 'wb'))

Network = Network.run_training(opts=opts)
