## HE cell classification pipeline 

Cells are trained using Supervised CNN and the ensemble classification determines the final output class probability of the single cells.

Cell class used for classification are epithelial cell, stromal cell, lymphocyte and other cell. The color codes are present in the `HE_Fib_Lym_Tum_Others.csv`.

Example cell annotations on tile images and respective csv for the slides are organised as in below hyperlink.
https://github.com/pathdata/UNMaSk/blob/master/HE_cell_classification/train/cell_annotations/AnnotatedTiles

https://github.com/pathdata/UNMaSk/tree/master/HE_cell_classification/train/cell_annotations/celllabels


## Docker container

### Docker image for GPU environment
Tensorflow GPU container -> docker://nrypri001docker/tf1p4:IHCv1                          

### Docker image for CPU environment
Tensorflow CPU container -> docker://nrypri001docker/tfcpu:HEv1

### Publicly accessible webpage for cpu environment
https://hub.docker.com/r/nrypri001docker/tfcpu


### Command line arguments for prediction

--------------------------------------------------------------------------------------------------------------------------

``` predict_Local.py --exp_dir=exp_dir --data_dir=data_dir --results_dir=results_dir     --detection_results_path=detection_results_path        --tissue_segment_dir=tissue_segment_dir --file_name_pattern=file_name_pattern ```

--------------------------------------------------------------------------------------------------------------------------

#exp_dir-> checkpoint_path                        
#data_dir-> cws_path                               
#result_dir-> classification result_path                                    
#detection_dir-> detection_path                                     
#tissue_segment_dir-> tissue_segmentation_result_path
#file_name_pattern -> *.svs(WSI slide extension obtained from the prefix or extension suffix)

##### Note
Please cite the article if the code is used completely or partially in your work.


