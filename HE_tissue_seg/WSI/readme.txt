Add the WSI images in this directory and run the code main_WSI.py with the respective command line arguments.

# --mode=WSI_test---> To directly run prediction on WSI image
# For main_WSI.py
# command line arguments example
# --model=model_HE_Inception_unet --test=Test --result=output --mode=WSI_test

Running tissue segmentation

usage: main.py [-h] [-model MODEL] [-test TEST] [-result RESULT] [-mode MODE]
               [-bs BATCH_SIZE] [-epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL, --model MODEL
                        path to the model
  -test TEST, --test TEST
                        path to test images
  -result RESULT, --result RESULT
                        path to predicted result images
  -mode MODE, --mode MODE
                        train or predict To perform prediction on test images
                        prediction mode is set to test
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size of training images
  -epochs EPOCHS, --epochs EPOCHS
                        total number of epochs

