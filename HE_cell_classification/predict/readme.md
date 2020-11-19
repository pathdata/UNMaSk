# HE Cell Classification pipeline in tensorflow1p4 environment

Cells are trained using Supervised CNN and the ensemble classification determines the final output class probability of the cell.

Cell class used for classification are epithelial cell, stromal cell, lymphocyte and other cell. The color codes are present in the `HE_Fib_Lym_Tum_Others.csv`.

Please reference the citation if the code is used completely or partially in your work.

# Docker container
Tensorflow GPU container -> docker://nrypri001docker/tf1p4:IHCv1
Tensorflow CPU container -> docker://nrypri001docker/tfcpu:HEv1
