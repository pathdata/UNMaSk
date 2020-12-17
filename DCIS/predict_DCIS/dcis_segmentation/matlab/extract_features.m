function feat = extract_features(I, features)

    feat = [];
    for iFeatures = 1:length(features)
        switch features{iFeatures}
            case 'rgb'
                feat = cat(3,feat,single(I));
            case 'lab'
                feat = cat(3,feat,single(rgb2lab(I)));
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
        end
    end
end