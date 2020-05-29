function [ Norm ] = NormReinhardSlide( Source, Slide, Reference, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NormRienhard: Normalise a Source image with respect to a Reference image
% using Reinhard's method.
% 
% This routine takes a Source and Reference image as input and uses Reinhard's 
% method (Reference below) to normalise the stain of Source image
% 
%
% Input:
% Source         - RGB Source image.
% Target         - RGB Reference image.
% verbose        - (optional) Display results.
%                  (default 0)
%
%
% Output:
% Norm           - Normalised RGB Source image.
%
%
% References:
% [1] E Reinhard, M Adhikhmin, B Gooch, P Shirley. "Color transfer between 
%     images". IEEE Computer Graphics and Applications, vol.21 no.5, pp.
%     34-41, 2001.
%
% Copyright (c) 2013, Adnan Khan
% Department of Computer Science,
% University of Warwick, UK.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('verbose', 'var') || isempty(verbose)
   verbose = 0;
%else
%    warning('Verbose mode for individual normalisation functions is likely to be removed in a future release. Please use Norm.m for Visualisation.');
end

if ~exist('Source', 'var') || isempty(Source)
   error('Please supply a Source Image.');
end

if ~exist('Reference', 'var') || isempty(Reference)
   error('Please supply a Reference Image.');
end

tissue = (Slide(:,:,1)./Slide(:,:,2))>1;
se = strel('disk',5,4);
tissue = imdilate(tissue, se);
% RGB to LAB colour space conversion for Source/Ref Image
SourceLab = applycform(im2double(Source), makecform('srgb2lab'));
SlideLab = applycform(im2double(Slide), makecform('srgb2lab'));
L=SlideLab(:,:,1); A=SlideLab(:,:,2); B=SlideLab(:,:,3);

% Means of Source image channels in Lab Colourspace
%ms = mean(reshape(SlideLab, [], 3));
ml = mean(L(tissue)); ma = mean(A(tissue)); mb = mean(B(tissue));
ms = [ml, ma, mb];

% Standard deviations of Source image channels in Lab Colourspace
%stds = std(reshape(SlideLab, [], 3));
sl = std(L(tissue)); sa = std(A(tissue)); sb = std(B(tissue));
stds = [sl, sa, sb];

% RGB to LAB colour space conversion for Target Image
TargetLab = applycform(im2double(Reference), makecform('srgb2lab'));

% Means of Target image channels in Lab Colourspace
mt = mean(reshape(TargetLab, [], 3));

% Standard deviations of Target image channels in Lab Colourspace
stdt = std(reshape(TargetLab, [], 3));


% Normalise each channel based on statistics of source and target images
NormLab(:,:,1) = ((SourceLab(:,:,1)-ms(1))*(stdt(1)/stds(1)))+mt(1);
NormLab(:,:,2) = ((SourceLab(:,:,2)-ms(2))*(stdt(2)/stds(2)))+mt(2);
NormLab(:,:,3) = ((SourceLab(:,:,3)-ms(3))*(stdt(3)/stds(3)))+mt(3);

% LAB to RGB conversion
Norm = applycform(NormLab, makecform('lab2srgb'));

% Display results if verbose mode is true
if verbose
    figure;
    subplot(1,3,1); imshow(Source);   title('Source Image');
    subplot(1,3,2); imshow(Reference);   title('Target Image');
    subplot(1,3,3); imshow(Norm); title('Normalised (Reinhard)');
end

end

        

   