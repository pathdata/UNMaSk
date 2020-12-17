
# Workflow of DCIS segmentation

### Prepare Data using TrainData

1. IM-Net used free hand annotation masks and the images and the data is saved in mat format

2. Convert the mat files to single tfrecord file on for training and one for validation with train/valid split of 0.7 before training




### Run prediction on the test images in predict_CIS directory using generate_output_DCIS.py script

1. Predict output on test images based on the inference file generated in exp_dir during training

### Run Spatial analysis after stitching the tiled output and generate Morisita score for each slide.
