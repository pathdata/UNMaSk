import sys

from dcis_segmentation.subpackages import NetworkOptions
from dcis_segmentation import save_output

#### checkpoint/model dir -> exp_dir #####
#### img_tile/tile dir    -> data_dir ####
#### output               -> result_subdir ####

if __name__ == '__main__':
    opts = NetworkOptions.NetworkOptions(exp_dir=r'ExpDir-CV1',
                                                 
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=508,
                                         image_width=508,
                                         label_height=508,
                                         label_width=508,
                                         in_feat_dim=3,
                                         in_label_dim=4,
                                         num_of_classes=2,
                                         batch_size=1,
                                         data_dir=r'img_tile',
                                         results_dir=r'results/'
                                                     'dcis_segmentation',
                                         current_epoch_num=0,
                                         file_name_pattern='*.ndpi',
                                         pre_process=False,
                                         result_subdir='20200501_TEST-TF1')

    if len(sys.argv) > 1:
        opts.data_dir = sys.argv[1]

    if len(sys.argv) > 2 and opts.sub_dir_name is None:
        try:
            opts.sub_dir_name = sys.argv[2]
        except NameError:
            opts.sub_dir_name = None

    save_output.run(opts_in=opts)
