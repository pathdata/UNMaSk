
##  DCIS segmentation workflow

### 1. Data preparation  

1. IM-Net uses free hand annotation masks collected exhaustively by ensuring no regions being missed in each represented tile.
2. Mat files are generated subsequently. 
3. Stride and patch size can be changed within the code depending on the network parameters.
4. Convert the mat files to single tfrecord file for training and validation.


### 2. Run prediction on the test images in predict_DCIS directory using generate_output_DCIS.py script

1. Predict segmentation on test images based on the inference file generated in exp_dir during training.
2. Details of docker and source codes on DCIS segmentation detailed in the repository (https://github.com/pathdata/UNMaSk/tree/master/DCIS/predict_DCIS)

### 3. Run spatial analysis after stitching the tiled output and generate Morisita score for each slide.

1. Voronoi maps are generated and the cell co-ordinates within each polygon is used to compute Morisita score between epithelial cell and lymphocyte.


#### Disclaimer
--------------------------------------------------------------------------------------------------------------------------

    The software is provided 'as is' with no implied fitness for purpose. 
    The author is exempted from any liability relating to the use of this software.  
    The software is provided for research use only. 
    The software is explicitly not licenced for re-distribution.
    Cite the article if the codes are reused in part/whole.

--------------------------------------------------------------------------------------------------------------------------




##### Notes
1. Data preparation and implementation codes are maintained in this repository and will be periodically updated. Please contact the corresponding authours for future collaboration and any queries regarding the implementation.

2. Due to file size restriction on github server all checkpoint files are not uploaded.
