function seg_to_detect_allDs(curr_WSI_mat_path, save_detection_csv_path, curr_WSI)

    Das = dir(fullfile(curr_WSI_mat_path, '*.mat'));
    for Da_n = 1:length(Das)
        curr_Da = Das(Da_n).name;
        if ~exist(fullfile(save_detection_csv_path, curr_WSI, [curr_Da(1:end-3), 'csv']), 'file')
            mat = load(fullfile(Das(Da_n).folder, curr_Da));
            if isfield(mat, 'mat')
                mat = mat.mat;
            end
            CC = bwconncomp(mat.BinLabel);
            S = regionprops(CC,'Centroid');
            centroids = round(cat(1, S.Centroid));
            V = cell(size(centroids,1),3);
            detection_table = cell2table(V);
            if ~isempty(centroids)
                detection_table.V1 = repmat({'None'},[size(centroids,1),1]);
                detection_table.V2 = centroids(:,1);
                detection_table.V3 = centroids(:,2);
                writetable(detection_table, fullfile(save_detection_csv_path, curr_WSI, [curr_Da(1:end-3), 'csv']));
            else
                fileID = fopen(fullfile(save_detection_csv_path, curr_WSI, [curr_Da(1:end-3), 'csv']), 'w');
                fprintf(fileID, 'V1,V2,V3');
                fclose(fileID);
            end
        end
    end
end