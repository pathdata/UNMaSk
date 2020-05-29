function pre_process_images(matlab_input)

output_path = matlab_input.output_path;
sub_dir_name = matlab_input.sub_dir_name;
tissue_segment_dir = matlab_input.tissue_segment_dir;
input_path = matlab_input.input_path;
features = matlab_input.feat;

if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name), 'dir')
    mkdir(fullfile(output_path, 'pre_processed', sub_dir_name));
end
if ~isempty(tissue_segment_dir)
    files_tissue = dir(fullfile(tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat'));
else
    files_tissue = dir(fullfile(input_path, 'Da*.jpg'));
end
display(matlab_input)
fprintf('Pre-Processing %s\n', sub_dir_name);
parfor i = 1:length(files_tissue)
    TargetImage = imread('TargetImage.png');
    if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'mat']), 'file')        
        I = imread(fullfile(input_path, [files_tissue(i).name(1:end-3), 'jpg']));
        I = im2uint8(NormReinhard( I, TargetImage));
        feat = extract_features(I, features);
        parsave(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'mat']), feat);
        fprintf('Saved %s\n', ...
            fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'mat']))
    else
        fprintf('Already Pre-Processed %s\n', ...
            fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'mat']))
    end
end