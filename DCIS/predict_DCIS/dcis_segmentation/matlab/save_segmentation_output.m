function save_segmentation_output(results_dir, sub_dir_name, mat_file_name, image_path_full)
    
    if ~exist(...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [mat_file_name(1:end-3), 'png']), 'file')
        mat = load(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name));
        if isfield(mat, 'mat')
            mat = mat.mat;
        end
        %[~,BinLabel] = max(mat.output, [],3);
        %mat.BinLabel = BinLabel>1;
        mat.BinLabel = mat.output(:,:,2)>0.2;
        mat.BinLabel = imfill(mat.BinLabel, 'holes');
        contour = edge(mat.BinLabel);
        im = imread(image_path_full);
        im = im2double(im);
		mask_rgb = repmat(mat.BinLabel, [1 1 3]);
		[r,c,h]=size(mask_rgb);
		imgrz=imresize(mask_rgb,[r/16,c/16]);
        im1 = im(:,:,1); im2 = im(:,:,2); im3 = im(:,:,3);
        im1(contour) = 0; im2(contour) = 1; im3(contour) = 0;
        annotated_image = cat(3, im1, im2, im3);
        imwrite(annotated_image, fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-4), '.png']), 'png');
		imwrite(im2double(mask_rgb), fullfile(results_dir, 'mask_image', sub_dir_name, [mat_file_name(1:end-3), 'png']), 'png');
		imwrite(im2double(imgrz), fullfile(results_dir, 'maskrz_image', sub_dir_name, [mat_file_name(1:end-3), 'png']), 'png');
        parsave_mat(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name),  mat);
        close all;
    else
        fprintf('Already Processed %s\n', ...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [mat_file_name(1:end-3), 'png']))
    end
end