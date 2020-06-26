function pre_process_images(matlab_input)

output_path = matlab_input.output_path;
sub_dir_name = matlab_input.sub_dir_name;
tissue_segment_dir = matlab_input.tissue_segment_dir;
input_path = matlab_input.input_path;
features = matlab_input.feat;
extra.tissue_seg_mat_dir = matlab_input.tissue_segment_dir;
if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name), 'dir')
    mkdir(fullfile(output_path, 'pre_processed', sub_dir_name));
end
if ~isempty(tissue_segment_dir)
    files_tissue = dir(fullfile(tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat'));
else
    files_tissue = dir(fullfile(input_path, 'Da*.jpg'));
end
parfor i = 1:length(files_tissue)
    if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']), 'file')
        fprintf('%s\n', fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']));
%         TargetImage = imread('Target.png');
        im = imread(fullfile(input_path, [files_tissue(i).name(1:end-3), 'jpg']));
        im = normalise_image(im, 1);        
        feat = get_feat(im, features, extra);
        h5save(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']), feat, 'feat');
    
    else
        fprintf('Already Pre-Processed %s\n', ...
            fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']))
    end
end
end