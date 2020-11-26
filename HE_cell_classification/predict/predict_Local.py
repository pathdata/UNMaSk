import os
#from parse_arguments import get_parsed_arguments

from classifier.sccnn_classifier import SccnnClassifier
from classifier.subpackages import NetworkOptions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################################################################################################

#exp_dir-> checkpoint_path
#data_dir-> cws_path
#result_dir-> classification result_path
#detection_dir-> detection_path
#tissue_segment_dir-> tissue_segmentation_result_path if available( this parameter is optional)


################################################################################################





opts = NetworkOptions.NetworkOptions(exp_dir=r'ExpDir-TA-Duke\checkpoint',#args.exp_dir
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=51,
                                     image_width=51,
                                     in_feat_dim=3,
                                     in_label_dim=1,
                                     num_of_classes=4,
                                     batch_size=100,
                                     data_dir=r'cws_DAVE',#args.data_dir
                                     results_dir=r'classification_results_HE', #args.results_dir
                                     detection_results_path=r'detection_DAVE', #args.detection_results_path
                                     tissue_segment_dir=r'detection_DAVE_TS',  #args.tissue_segment_dir
                                     preprocessed_dir=None,
                                     current_epoch_num=0,
                                     file_name_pattern='*.svs', #args.file_name_pattern
                                     pre_process=False,
                                     color_code_file='HE_Fib_Lym_Tum_Others.csv')





opts.results_dir = (os.path.join(opts.results_dir, '2020ENS_TA_DUKE_HE_TEST'))


if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir)
if not os.path.isdir(os.path.join(opts.results_dir, 'mat')):
    os.makedirs(os.path.join(opts.results_dir, 'mat'))
if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
    os.makedirs(os.path.join(opts.results_dir, 'csv'))


Network = SccnnClassifier(batch_size=opts.batch_size,
                                           image_height=opts.image_height,
                                           image_width=opts.image_width,
                                           in_feat_dim=opts.in_feat_dim,
                                           in_label_dim=opts.in_label_dim,
                                           num_of_classes=opts.num_of_classes)
#print(opts)
Network.generate_output(opts=opts)
