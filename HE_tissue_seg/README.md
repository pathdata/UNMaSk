# UNMaSk: Unmasking the immune microecology of ductal carcinoma in situ with deep learning.


# Parameters of train and predict for tissue segmentation pipeline

Command line arguments for Training

```python main.py --batch_size=4 --epochs=100 --mode=train```

Command line arguments for Prediction

`python main.py --model=model_HE_Inception_unet --test=TestImages --result=output --mode=test`

--mode=WSI_test---> To directly run prediction on WSI image
For main_WSI.py
command line arguments example


`python main_WSI.py --model=model_HE_Inception_unet --test=WSI --result=output --mode=WSI_test`

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





# Citation

# Reference

All training data of carcinoma in situ regions that were annotated as a part of the project is made available in this github repository.
Training data tiles were anonymised from raw HE image tiles. Request for data access for the Duke samples can be submitted to E.S.H and Y.Y

# Training
Data preparation and implementation codes are maintained in this repository and will be periodically updated. Please contact the corresponding authours for future colloborations and any queries regarding the implementation.

