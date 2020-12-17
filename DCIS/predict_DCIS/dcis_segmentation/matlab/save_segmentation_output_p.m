function save_segmentation_output_p(results_dir, sub_dir_name, image_path)
    if ~exist(fullfile(results_dir, 'annotated_images'), 'dir')
        mkdir(fullfile(results_dir, 'annotated_images'));
    end
	if ~exist(fullfile(results_dir, 'mask_image'), 'dir')
        mkdir(fullfile(results_dir, 'mask_image'));
    end
	if ~exist(fullfile(results_dir, 'maskrz_image'), 'dir')
        mkdir(fullfile(results_dir, 'maskrz_image'));
    end
    if ~exist(fullfile(results_dir, 'annotated_images', sub_dir_name), 'dir')
        mkdir(fullfile(results_dir, 'annotated_images', sub_dir_name));
    end
	if ~exist(fullfile(results_dir, 'mask_image',sub_dir_name), 'dir')
        mkdir(fullfile(results_dir, 'mask_image',sub_dir_name));
    end
	if ~exist(fullfile(results_dir, 'maskrz_image',sub_dir_name), 'dir')
        mkdir(fullfile(results_dir, 'maskrz_image',sub_dir_name));
    end
    files = dir(fullfile(results_dir, 'mat', sub_dir_name, 'Da*.mat'));
    warning('off');
    parfor i = 1:length(files)
        fprintf('%s\n', files(i).name);
        mat_file_name = files(i).name;
        image_path_full = fullfile(image_path, [files(i).name(1:end-3), 'jpg']);
        save_segmentation_output(results_dir, sub_dir_name, mat_file_name, image_path_full);
    end
end