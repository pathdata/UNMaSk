
# Workflow of DCIS segmentation

### Prepare Data using TrainData

1. IM-Net used free hand annotation masks and the images in mat format

2. Convert the mat to tfrecords before training

3. Predict the output based on the inference file generated in exp_dir


### Run prediction on the test images in predict_CIS directory using generate_output_DCIS.py script

### Run Spatial analysis after stitching the tiled output and generate Morisita score for each slide.
