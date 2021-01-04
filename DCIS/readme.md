
# Workflow of DCIS segmentation

### 1. Data preparation  

1. IM-Net uses free hand annotation masks collected exhaustively by ensuring no regions being missed in each represented tile.
2. Mat files are generated subsequently. 
3. Stride and patch size can be changed within the code depending on the network parameters.
4. Convert the mat files to single tfrecord file for training and validation.


### 2. Run prediction on the test images in predict_CIS directory using generate_output_DCIS.py script

1. Predict output on test images based on the inference file generated in exp_dir during training

### 3. Run spatial analysis after stitching the tiled output and generate Morisita score for each slide.

1. Voronoi maps are generated and the cell co-ordinates within each polygon is used to compute Morisita score between epithelial cell and lymphocyte.





##### Notes
1. Data preparation and implementation codes are maintained in this repository and will be periodically updated. Please contact the corresponding authours for future collaboration and any queries regarding the implementation.

2. Due to file size restriction on github server all checkpoint files are not uploaded.
