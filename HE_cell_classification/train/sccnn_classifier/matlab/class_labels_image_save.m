clc
clear
close all;
%%
class_labels_dir = 'R:\tracerx\tracerx100\IHC_diagnostic\Misc\Annotations-180308\cell_labels-180308\celllabels';
cws_dir = 'T:\tracerx100\IHC_diagnostic\data\cws';
strength = 3;
color_code_file = 'IHC_CD4_CD8_FoxP3.csv';
save_dir = 'R:\tracerx\tracerx100\IHC_diagnostic\Misc\Annotations-180308\cell_labels-180308\annotated_image';
%%
wsi_dirs = dir(fullfile(class_labels_dir, '*.ndpi'));

for wsi_n = 1:length(wsi_dirs)
   files = dir(fullfile(wsi_dirs(wsi_n).folder, wsi_dirs(wsi_n).name, '*.csv'));
   for files_n = 1:length(files)
      file_name = files(files_n).name(1:end-4);
      if exist(fullfile(cws_dir, wsi_dirs(wsi_n).name, [file_name, '.jpg']), 'file')
          im = imread(fullfile(cws_dir, wsi_dirs(wsi_n).name, [file_name, '.jpg']));
          class_table = readtable(fullfile(class_labels_dir, wsi_dirs(wsi_n).name, files(files_n).name));
          colorcodes = readtable(fullfile('colorcodes', color_code_file));
          detection = [class_table.V2, class_table.V3];
          class = class_table.V1;
          discard = detection(:,1)<1;
          class(discard, :) = [];
          detection(discard, :) = [];
          discard = detection(:,2)<1;
          detection(discard, :) = [];
          class(discard, :) = [];
          discard = detection(:,1)>2000;
          class(discard, :) = [];
          detection(discard, :) = [];
          discard = detection(:,2)>2000;
          detection(discard, :) = [];
          class(discard, :) = [];
          for c = 1:height(colorcodes)-1
            im = annotate_image_with_class(im, detection(strcmp(class, colorcodes.class{c}),:), ...
                hex2rgb(colorcodes.color{c}), strength);
%             if ~exist(fullfile(save_dir, wsi_dirs(wsi_n).name), 'dir')
%                 mkdir(fullfile(save_dir, wsi_dirs(wsi_n).name));
%             end
          end
            imwrite(im, fullfile(save_dir, [wsi_dirs(wsi_n).name(1:end-4), '_', file_name, '.jpg']));
      end
   end
end