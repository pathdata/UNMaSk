function out_im = normalise_image(im, normalisation)

switch normalisation
    case 0
        I = im2uint8(im);
    case 1
        I = im2uint8(Retinex(im));         % adjust using Retinex
    case 2
        I = Retinex(im);         % adjust using Retinex
        TargetImage = imread('Target.png');
        [ I ] = im2uint8(NormReinhard( I, TargetImage));
    case 3
        TargetImage = imread('Target.png');
        [ I ] = im2uint8(NormReinhard( im, TargetImage));
end

out_im = I;    
end