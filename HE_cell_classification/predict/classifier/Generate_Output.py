import os

import sccnn_classifier as sccnn_classifier
from classifier.subpackages import NetworkOptions

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = NetworkOptions.NetworkOptions(exp_dir='D:\\August2018\\cell_classification_HE\\20171019-SCCNNClassifier_TA\\ExpDir-Duke',
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=51,
                                     image_width=51,
                                     in_feat_dim=3,
                                     in_label_dim=1,
                                     num_of_classes=4,
                                     batch_size=100,
                                     data_dir='D:\\August2018\\cell_classification_HE\\data\\cws',
                                     results_dir='D:\\August2018\\cell_classification_HE\\classification_TA_DL_results',
                                     detection_results_path='D:\\August2018\\cell_classification_HE\\results\\20180810',
                                     tissue_segment_dir='D:\\August2018\\cell_classification_HE\\tissue_segmentation',
                                     preprocessed_dir=None,
                                     current_epoch_num=0,
                                     file_name_pattern='*.ndpi',
                                     pre_process=True,
                                     color_code_file='HE_Fib_Lym_Tum_Others.csv')

opts.results_dir = (os.path.join(opts.results_dir, '20171019_TA'))
opts.preprocessed_dir = os.path.join(opts.preprocessed_dir, '20171019_TA')

if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir)
if not os.path.isdir(os.path.join(opts.results_dir, 'mat')):
    os.makedirs(os.path.join(opts.results_dir, 'mat'))
if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
    os.makedirs(os.path.join(opts.results_dir, 'csv'))
if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed')):
    os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed'))

Network = sccnn_classifier.SccnnClassifier(batch_size=opts.batch_size,
                                           image_height=opts.image_height,
                                           image_width=opts.image_width,
                                           in_feat_dim=opts.in_feat_dim,
                                           in_label_dim=opts.in_label_dim,
                                           num_of_classes=opts.num_of_classes)
Network.generate_output(opts=opts)
