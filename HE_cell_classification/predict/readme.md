# HE Cell Classification pipeline in tensorflow1p4 environment

Cells are trained using Supervised CNN and the ensemble classification determines the final output class probability of the cell.

Cell class used for classification are epithelial cell, stromal cell, lymphocyte and other cell. The color codes are present in the `HE_Fib_Lym_Tum_Others.csv`.

Please reference the citation if the code is used completely or partially in your work.

# Docker container

# Docker image for GPU environment
Tensorflow GPU container -> docker://nrypri001docker/tf1p4:IHCv1                          

# Docker image for CPU environment
Tensorflow CPU container -> docker://nrypri001docker/tfcpu:HEv1

# Publicly accessible webpage for cpu environment
https://hub.docker.com/r/nrypri001docker/tfcpu

# Parameters for prediction of classification

#exp_dir-> checkpoint_path                        
#data_dir-> cws_path                               
#result_dir-> classification result_path                                    
#detection_dir-> detection_path                                     
#tissue_segment_dir-> tissue_segmentation_result_path
#file_name_pattern -> *.svs(WSI slide extension obtained from the prefix or extension suffix)

``` predict_Local.py --exp_dir=exp_dir --data_dir=data_dir --results_dir=results_dir     --detection_results_path=detection_results_path --tissue_segment_dir=tissue_segment_dir -file_name_pattern=file_name_pattern ```
