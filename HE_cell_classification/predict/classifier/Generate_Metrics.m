clc
clear
close all
%%
files = dir('ExpDir\output\*.mat');

group = [];
grouphat = [];
for i = 1:length(files)
    in = load(['ExpDir\output\',files(i).name]);
    C = unique(in.cell_ids);
    L = zeros(length(C),1);
    P = zeros(length(C),1);
    for j = 1:length(C)
        labelsC = in.labels(in.cell_ids==C(j));
        switch labelsC(1)
            case 'e'
                L(j) = 1;
            case 'f'
                L(j) = 1;
            case 'l'
                L(j) = 2;
            case 'p'
                L(j) = 2;
            case 't'
                L(j) = 3;
            otherwise
                L(j) = 4;
        end
        P(j) = mode(in.output(in.cell_ids==C(j)));
    end
    group = [group;L]; %#ok<*AGROW>
    grouphat = [grouphat;P];
    
    clear in;
end

stats = confusionmatStats(group,grouphat);

