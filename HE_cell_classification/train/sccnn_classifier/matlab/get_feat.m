function feat = get_feat(I, features, extra)

feat = [];
    for iFeatures = 1:length(features)
        switch features{iFeatures}
            case 'rgb'
                feat = cat(3,feat,single(I)./255);
            case 'lab'
                feat = cat(3,feat,single(rgb2lab(im2double(I))));
            case 'h'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ H ] = PseudoColourStains( DCh, colourMat );
                H = rgb2gray(H);
                feat = cat(3,feat,single(H));
            case 'e'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ ~, E ] = PseudoColourStains( DCh, colourMat );
                E = rgb2gray(E);
                feat = cat(3,feat,single(E));
            case 'he'
                doubleRGB = RemoveArtefact(I);
                colourMat = EstUsingSCD(doubleRGB);
                [ DCh ] = Deconvolve( I, colourMat );
                [ H, E ] = PseudoColourStains( DCh, colourMat );
                H = rgb2gray(H);
                E = rgb2gray(E);
                feat = cat(3,feat,single(H),single(E));
            case 'br'
                BR = BlueRatioImage(I);
                feat = cat(3,feat,single(BR));
            case 'grey'
                grey = rgb2gray(I);
                feat = cat(3,feat,single(grey));
            case 'tissue_seg'
                mat = load(fullfile(extra.tissue_seg_mat_dir, extra.curr_mode, extra.curr_wsi, [extra.jpg_name, '.mat']));
                if isfield(mat, 'mat')
                    mat = mat.mat;
                end
                temp_feat = imresize(mat.output(:,:,2), size(I(:,:,1)));
                feat = cat(3, feat, single(temp_feat));
            case 'cell_seg'
                mat = load(fullfile(extra.cell_seg_mat_dir, extra.curr_mode, extra.curr_wsi, [extra.jpg_name, '.mat']));
                if isfield(mat, 'mat')
                    mat = mat.mat;
                end
                feat = cat(3, feat, single(mat.output(:,:,2)));
        end
    end

end