function[ ] = get_duct_segm_local(  )
%Create whole-slide duct segmentation mask and ecologial ROI

imagesPath='DCIS_Duke/Data/cws/';                %cws
cellClassPath = 'DL_Class_Results/20180131/csv/';%cellclassification_csv
ductSegmPath = 'DL_DuctSegm/20180104/mat/';      %mat
outputPath = 'Duct_Eco/Duct_Segm/';              %output

folders = dir(fullfile(imagesPath, 'DCIS*'));
isub = [folders(:).isdir];      
folders = {folders(isub).name}';
folders(ismember(folders,{'.','..'})) = [];

duct_nbh_size = 10;
scale = 16;
tile_scaled_size = 2000/scale;
se = strel('disk',5);

for i=1:length(folders)
    fprintf('Processing %s \n', folders{i})
    load(fullfile(imagesPath, folders{i}, 'param.mat'));
    tiles_ids = reshape(1:ceil(slide_dimension(1)/2000)* ceil(slide_dimension(2)/2000), ...
        [ceil(slide_dimension(1)/2000), ceil(slide_dimension(2)/2000)]);
    tiles_ids = tiles_ids.'-1;
    slide_mask = zeros((size(tiles_ids,1)-1)*tile_scaled_size, (size(tiles_ids,2)-1)*tile_scaled_size);
    for c = 1:size(tiles_ids,2)
        for r = 1:size(tiles_ids,1)
            fprintf('Processing %d tile \n', tiles_ids(r,c))
            mat=load(fullfile(ductSegmPath, folders{i}, strcat('Da', num2str(tiles_ids(r,c)), '.mat')));
            if isfield(mat, 'mat')
                da_mask=mat.mat.BinLabel;
            else
                da_mask=mat.output(:,:,1)<0.5;
            end
            da_mask = imopen(da_mask,se);
            da_mask = imclose(da_mask,se);
            
            %Remove ducts with too few cancer cell detections
            filename = fullfile(cellClassPath,folders{i}, strcat('Da', num2str(tiles_ids(r,c)), '.csv'));
            delimiter = ',';
            startRow = 2;
            formatSpec = '%q%f%f%[^\n\r]';
            fileID = fopen(fullfile(filename),'r');
            if fileID~=-1
                dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
                    'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
           
            fclose(fileID);
            dataArray([2, 3]) = cellfun(@(x) num2cell(x), dataArray([2, 3]), 'UniformOutput', false);
            DaCellPos = [dataArray{1:end-1}];
            Da_t = DaCellPos(strcmp('t',DaCellPos(:,1)),:);
            cc_da = bwconncomp(da_mask);
            for cc=1:cc_da.NumObjects 
                count=0;
                for tc = 1:size(Da_t,1)
                    cellInd = sub2ind(size(da_mask), Da_t{tc,3}, Da_t{tc,2});
                    if ~isempty(find(cc_da.PixelIdxList{1,cc}==cellInd, 1))
                        count = count+1;
                    end
                end
                if count < 5
                    da_mask(cc_da.PixelIdxList{1,cc}) = 0;
                end
            end
            
            else 
                da_mask = zeros(size(da_mask));
            end
            
            tile_m = imresize(da_mask, 1/scale);
            slide_mask(((r-1)*tile_scaled_size+1):((r-1)*tile_scaled_size+size(tile_m,1)), ...
                ((c-1)*tile_scaled_size+1):((c-1)*tile_scaled_size+size(tile_m,2)))=tile_m;
            
        end
    end

    D = bwdist(slide_mask);
    L = watershed(D);
    L(bwdist(slide_mask)>duct_nbh_size) = 0;
    
    save(fullfile(outputPath, strcat(folders{i}, '_ducts.mat')), 'slide_mask', 'L')
    %rgb = label2rgb(L,'jet',[.5 .5 .5]);
    
end
end