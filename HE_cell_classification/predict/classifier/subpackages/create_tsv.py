import os


def create_tsv(labels_all, tsv_save_path):
    with open(os.path.join(tsv_save_path, 'metadata.tsv'), 'w') as f:
        f.write("Index\tClassName\tLabel\tLabel_4Class\n")
        for index, label in enumerate(labels_all):
            label_4class = 'others'
            class_name = 5

            if label == 'e':
                label_4class = 'fibroblasts'
                label = 'endothelium'
                class_name = 1
            if label == 'f':
                label_4class = 'fibroblasts'
                label = 'fibroblasts'
                class_name = 2
            if label == 'l':
                label_4class = 'lymphocytes'
                label = 'lymphocytes'
                class_name = 3
            if label == 'p':
                label_4class = 'lymphocytes'
                label = 'plasma_cells'
                class_name = 4
            if label == 't':
                label_4class = 'tumour'
                label = 'tumour'
                class_name = 5
            if label == 'm':
                label = 'macrophages'
                class_name = 6
            if label == 'o':
                label = 'eosinophils'
                class_name = 7
            if label == 'r':
                label = 'respiratory_epithelium'
                class_name = 8
            if label == 'c':
                label = 'cartilage'
                class_name = 9
            f.write("%d\t%d\t%s\t%s\n" % (index, class_name, label, label_4class))
