## UNMaSk: Unmasking the immune microecology of ductal carcinoma in situ with deep learning.

## Tissue Segmentation

This pipeline will delineate the tissue from the background. 

## Tissue segmentation module tree

 * [Tissue Segmentation](./HE_tissue_seg)
 * [TestImages](./TestImages)
    * [TA008_027-2015-12-07_18.23.05_Ss1.jpg](./TestImages/TA008_027-2015-12-07_18.23.05_Ss1.jpg)
    * [TA022_006-2015-11-23_14.21.23_Ss1.jpg](./TestImages/TA022_006-2015-11-23_14.21.23_Ss1.jpg)
 * [Train_HE](./Train_HE)
    * [he_imgs_train.npy](./Train_HE/he_imgs_train.npy)
    * [he_imgs_mask_train.npy](./Train_HE/he_imgs_mask_train.npy)
    * [imgs_mask_valid.npy](./Train_HE/imgs_mask_valid.npy)
    * [imgs_valid.npy](./Train_HE/imgs_valid.npy)
 * [scripts](./scripts)
   * [predict_HE_Inception_unet.py](./scripts/predict_HE_Inception_unet.py)
   * [predict_parser.py](./scripts/predict_parser.py)
   * [slidingpatches.py](./scripts/slidingpatches.py)
   * [train_HE_BLInception_unet.py](./scripts/train_HE_BLInception_unet.py)
 * [model](./model_HE_Inception_unet)
    * [model-tissue-seg.h5](./model_HE_Inception_unet/model-tissue-seg.h5)    
 * [WSI](./WSI)
    * [HE_S.svs](./WSI/HE_S.svs)
 * [main.py](./main.py)
 * [main_WSI.py](./main_WSI.py)
 * [requirements.txt](./requirement.txt)
 * [Dockerfile](./Dockerfile)
 * [README.md](./README.md)
 
 
## Docker Container

```docker://nrypri001docker/tf:tsv1 ```

## Publicly accessible weblink

https://hub.docker.com/repository/docker/nrypri001docker/tf

### Parameters used for prediction of tissue segmentation on test images

--------------------------------------------------------------------------------------------------------------------------

Command line arguments for Prediction

` main.py --model=model_HE_Inception_unet --test=TestImages --result=output `

--mode=WSI_test---> To directly run prediction on WSI image
For main_WSI.py
command line arguments example


` main_WSI.py --model=model_HE_Inception_unet --test=WSI --result=output --mode=WSI_test`

--------------------------------------------------------------------------------------------------------------------------

Command line arguments for Training

``` main.py --batch_size=4 --epochs=100 --mode=train```



Running tissue segmentation
usage: main.py [-h] [-model MODEL] [-test TEST] [-result RESULT] [-mode MODE]
               [-bs BATCH_SIZE] [-epochs EPOCHS]

optional arguments:

  `-h, --help                          show this help message and exit`
  
  `-model MODEL, --model MODEL         path to the model`
                        
  `-test TEST, --test TEST             path to test images`
                        
  `-result RESULT, --result RESULT     path to predicted result images`
                        
  `-mode MODE, --mode MODE             train or predict To perform prediction on test images prediction mode called by setting mode flag to test`
                        
  `-bs BATCH_SIZE, --batch_size BATCH_SIZE    batch size of training images`
                        
  `-epochs EPOCHS, --epochs EPOCHS            total number of epochs`


## Citation


## Training
Images and respective mask used for tissue segmentation can be found as in the hyperlink below
https://github.com/pathdata/UNMaSk/tree/master/HE_Data/TransATAC/TA_orig
https://github.com/pathdata/UNMaSk/tree/master/HE_Data/TransATAC/TA_mask

Data preparation and implementation codes are maintained in this repository and will be periodically updated. Please contact the corresponding authours for future colloborations and any queries regarding the implementation.

#### Note
Source code is available in this repository and tested with the packaged docker image for tissue segmentation.

