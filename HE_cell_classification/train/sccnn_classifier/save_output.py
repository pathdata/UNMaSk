import os

from sccnn_classifier import sccnn_classifier


def none_str(s):
    return '' if s is None else str(s)


def run(opts_in):
    if opts_in.preprocessed_dir == opts_in.results_dir:
        opts_in.preprocessed_dir = os.path.join(opts_in.preprocessed_dir, opts_in.results_subdir)
    opts_in.results_dir = os.path.join(opts_in.results_dir, opts_in.results_subdir)

    if not os.path.isdir(opts_in.results_dir):
        os.makedirs(opts_in.results_dir, exist_ok=True)
    if not os.path.isdir(os.path.join(opts_in.results_dir, 'mat')):
        os.makedirs(os.path.join(opts_in.results_dir, 'mat'), exist_ok=True)
    if not os.path.isdir(os.path.join(opts_in.results_dir, 'annotated_images')):
        os.makedirs(os.path.join(opts_in.results_dir, 'annotated_images'), exist_ok=True)
    if not os.path.isdir(os.path.join(opts_in.results_dir, 'class_csv')):
        os.makedirs(os.path.join(opts_in.results_dir, 'class_csv'), exist_ok=True)
    if not os.path.isdir(os.path.join(opts_in.preprocessed_dir, 'pre_processed')):
        os.makedirs(os.path.join(opts_in.preprocessed_dir, 'pre_processed'), exist_ok=True)

    network = sccnn_classifier.SccnnClassifier(batch_size=opts_in.batch_size,
                                               image_height=opts_in.image_height,
                                               image_width=opts_in.image_width,
                                               in_feat_dim=opts_in.in_feat_dim,
                                               in_label_dim=opts_in.in_label_dim,
                                               num_of_classes=opts_in.num_of_classes)

    print('---------------------------------------------------------------', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print('opts.data_dir:' + os.path.join(opts_in.data_dir, none_str(opts_in.sub_dir_name)), flush=True)
    print('opts.results_dir:' + os.path.join(opts_in.results_dir, none_str(opts_in.sub_dir_name)), flush=True)
    print('opts.detection_results_path:' + os.path.join(
        opts_in.detection_results_path, none_str(opts_in.sub_dir_name)), flush=True)
    print('opts.preprocessed_dir:' + os.path.join(
        opts_in.preprocessed_dir, 'pre_processed', none_str(opts_in.sub_dir_name)), flush=True)
    print('opts.tissue_segmentation:' + os.path.join(
        opts_in.tissue_segment_dir, none_str(opts_in.sub_dir_name)), flush=True)
    print('opts.file_name_pattern:' + opts_in.file_name_pattern, flush=True)
    print('opts.pre_process:' + str(opts_in.pre_process), flush=True)
    print('opts.exp_dir:' + opts_in.exp_dir, flush=True)
    print('opts.color_code_file:' + opts_in.color_code_file, flush=True)
    print('---------------------------------------------------------------\n', flush=True)

    if opts_in.sub_dir_name is None:
        network.generate_output(opts=opts_in)
    else:
        network.generate_output_sub_dir(opts=opts_in, sub_dir_name=opts_in.sub_dir_name)
