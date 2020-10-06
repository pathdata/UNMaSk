import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_dir', '--exp_dir', dest='exp_dir', help='checkpoint parent directory path')
    parser.add_argument('-data_dir', '--data_dir', dest='data_dir', help='directory of cws/tile images of a slide')
    parser.add_argument('-results_dir', '--results_dir', dest='results_dir', help='directory to save prediction output')
    parser.add_argument('-detection_results_path','--detection_results_path', dest='detection_results_path', help='directory of csv containing detection results')
    parser.add_argument('-preprocess','--preprocessed_dir', dest='preprocessed_dir', help='directory of preprocessed images saved under result directory')
    parser.add_argument('-tissue_segment_dir','--tissue_segment_dir', dest='tissue_segment_dir', help='directory of tissue segmentation containing mat and this is optional parameter')
    parser.add_argument('-file_name_pattern','--file_name_pattern', dest='file_name_pattern', help='file extension pattern or slide name pattern')
    parser.add_argument('-cluster','--cluster', dest='cluster', help='flag to check if running on server or local', default=False)
    parser.add_argument('-num_examples_per_epoch_train', '--num_examples_per_epoch_train', type=int, dest='num_examples_per_epoch_train', help='num_examples_per_epoch_train', default=1)
    parser.add_argument('-num_examples_per_epoch_valid','--num_examples_per_epoch_valid', type=int, dest='num_examples_per_epoch_valid', help='num_examples_per_epoch_valid', default=1)
    parser.add_argument('-image_height', '--image_height', type=int, dest='image_height', help='image patch height', default=51)
    parser.add_argument('-image_width', '--image_width', type=int, dest='image_width', help='image patch width', default=51)
    parser.add_argument('-in_feat_dim', '--in_feat_dim', type=int, dest='in_feat_dim', help='image patch channel', default=3)
    parser.add_argument('-in_label_dim', '--in_label_dim', type=int, dest='in_label_dim', help='label_dimension', default=1)
    parser.add_argument('-num_of_classes', '--num_of_classes', type=int, dest='num_of_classes', help='num_of_classes', default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, dest='batch_size', help='batch_size', default=100)
    parser.add_argument('-num_of_epoch', '--num_of_epoch', type=int, dest='num_of_epoch', help='num_of_epoch', default=300)

    args = parser.parse_args()

    return args