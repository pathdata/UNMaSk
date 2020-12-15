# UNMaSk: Unmasking the immune microecology of ductal carcinoma in situ with deep learning.

Note: This project is a work in progress. 

UNMaSk pipeline consists of processing pipelines for both HE and IHC images.
1. Tissue Segmentation
2. Cell Detection
3. Cell Classification
4. Ductal carcinoma in situ Segmentation (organised in CIS)

Each of these pipelines are organised inside individual directory and you will be able to find more details in the respective sub-directories. Wherever possible docker images and command line instructions are specified to make it friendly for off the shelf users.

# Overview schematic of UNMaSk pipeline for DCIS segmentation.
<p align="center">
  <img src="environment/Fig1_overview.png" width="450" height="450"/>
 </p>
 
 # Schematic of IM-Net architecture for DCIS segmentation and schematic of DRDIN cell detection network. 
 <p align="center">
 
  <img src="environment/Fig2_ab_Revised_v1.png" width="450" height="450"/>
  </p>

# Training Data

Images used for training
https://github.com/pathdata/HE_Tissue_Segmentation/tree/master/CIS/TrainData

Ground truth images
https://github.com/pathdata/HE_Tissue_Segmentation/tree/master/CIS/TrainData/mask

Overlay of groundtruth on the training image
https://github.com/pathdata/HE_Tissue_Segmentation/tree/master/CIS/TrainData/overlay

# Illustrative images used in training IM-NET

<p align="center">
  
  <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_img_movie_001.gif" width="250" height="250"/>
  <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_mask_movie_001.gif" width="250" height="250"/>
  <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_overlay_movie_001.gif" width="250" height="250"/>
</p>

# Citation

# Reference

All training data of carcinoma in situ regions that were annotated as a part of the project is made available in this github repository.
Training data tiles were anonymised from raw HE image tiles. Request for data access for the Duke samples can be submitted to E.S.H and Y.Y

# Training
Data preparation and implementation codes are maintained in this repository and will be periodically updated. Please contact the corresponding authours for future collaboration and any queries regarding the implementation.

