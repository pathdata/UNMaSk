import os
import pathlib

from sccnn_classifier.subpackages import NetworkOptions
from sccnn_classifier import train


if os.name=='nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    save_tf_path = pathlib.Path(r'D:\2020_SCCNN\HE\tfrecords')
    train_tf_filename = 'TrainData-HE-2019-11'
    valid_tf_filename = 'ValidData-HE-2019-11'

    opts = NetworkOptions.NetworkOptions(exp_dir=os.path.normpath(os.path.join(os.getcwd(), 'SCCNN')),
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=51,
                                         image_width=51,
                                         in_feat_dim=3,
                                         in_label_dim=1,
                                         num_of_classes=4,
                                         batch_size=100,
                                         num_of_epoch=300,
                                         data_dir=save_tf_path,
                                         train_data_filename=train_tf_filename,
                                         valid_data_filename=valid_tf_filename,
                                         current_epoch_num=0)


    train.run(opts_in=opts)
