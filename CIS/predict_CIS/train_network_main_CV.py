import os
import pathlib

from dcis_segmentation.subpackages import NetworkOptions
from dcis_segmentation import train


if __name__ == '__main__':
    opts = NetworkOptions.NetworkOptions(exp_dir=os.path.normpath(os.path.join(os.getcwd(), r'ExpDir_DCIS_CV1_ADM_10E')),
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=600,
                                         image_width=600,
                                         label_height=600,
                                         label_width=600,
                                         crop_height=508,
                                         crop_width=508,
                                         in_feat_dim=3,
                                         in_label_dim=4,
                                         num_of_classes=2,
                                         batch_size=1,
                                         num_of_epoch=10,
                                         data_dir=str(pathlib.Path(r'/mnt/pnarayan/tfrecord_CV1')),
                                         train_data_filename='TrainData-HE-1911-CV1',
                                         valid_data_filename='ValidData-HE-1911-CV1',
                                         current_epoch_num=0)

    train.run(opts_in=opts)
