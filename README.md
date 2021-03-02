# UNMaSk: Unmasking the immune microecology of ductal carcinoma in situ with deep learning.

### Background
Ductal carcinoma in situ (DCIS) is a non-obligatory precursor of invasive ductal carcinoma (IDC). It is the most common mammographically detected breast cancer, however, predicting DCIS progression to IDC remains a major clinical challenge. A recent study has categorised DCIS evolution to IDC into four models, highlighting its heterogeneity. The evolutionary potential of individual DCIS ductules/ducts may dramatically differ, determined by not only their genetic mutations but also microenvironmental
selective pressure. However, given the complex spatial ductule structure, ecological dynamics between individual DCIS ducts and their surrounding microenvironment are difficult to measure by eye. These ultimately limits our ability to study the influence of the microenvironment on tumour evolution and progression.

### Aims 
Our primary aims were: (1) to develop and validate a computational pipeline that accurately detects and segments individual DCIS ducts; (2) to characterise the immune microecology for each DCIS duct using spatial statistics on H&E and IHC for TILs; (3) to test the difference in DCIS microecology between samples with pure DCIS
and DCIS samples derived from IDC patients (adjacent DCIS, as a surrogate for poor prognosis DCIS).


UNMaSk pipeline consists of processing pipelines for segmentation of ductal carcinoma in situ (DCIS) regions from whole slide images. The primary steps are broken down into four modules and are as given below.


### UNMaSk DCIS segmentation module tree
 * [HE_tissue_segmentation](./HE_tissue_seg)

 * [DCIS segmentation](./DCIS/predict_DCIS)
 
 * [HE_cell_detection](./HE_cell_detection)
 
 * [HE_cell_classification](./HE_cell_classification/predict)
    


Modular arrangement help in navigation to each module and supports both tile level and tissue level processing. Each of these pipelines are organised inside individual directory and you will be able to find detailed explanation in the respective sub-directories. Wherever possible docker images and command line instructions are specified to make it user friendly for off the shelf users.

## UNMaSk pipeline overview and architecture 


<p align="center">
  <img src="environment/gt_Fig1.png" width="350" height="250"/>
   <figcaption> Overview schematic of UNMaSk pipeline for DCIS segmentation UNet architecture for tissue segmentation and one
   of the existing deep learning methods, single-shot detector (SSD) architecture, used for DCIS detection.
 </p>
 
 <p align="center">
 <img src="environment/gt_Fig2.png" width="550" height="150"/>
<figcaption> Spatial Voronoi tessellation to examine local tissue ecology for each DCIS duct, based on deep learning results on DCIS segmentation and single-cell classification. Examples shown are immune depleted and immune predominant/inflamed ecology local to individual DCIS ducts from the DCIS immune colocalisation/Morisita Score (MS) spatial analysis. </figcaption>
 </p>
 
<p align="center">
  
   <img src="environment/Fig2_a.png" width="350" height="200"/>
   <figcaption> Schematic of IM-Net architecture for DCIS segmentation. (Note: High resolution image can be visualised by clicking on the above image)     
      </figcaption>
 </p>
 
 <p align="center">
    
   <img src="environment/Fig2_b.png" width="250" height="50"/>
   
   <figcaption> Schematic of DRDIN cell detection network. (Note: High resolution image can be visualised by clicking on the above image)
       </figcaption>
 
 </p>
 

  


### Training data and annotations used in DCIS segmentation

a. Images used for training
https://github.com/pathdata/UNMaSk/tree/master/DCIS/TrainData (Left)

b. Ground truth images
https://github.com/pathdata/UNMaSk/tree/master/DCIS/TrainData/mask (Middle)

c. Overlay of groundtruth on the training image
https://github.com/pathdata/UNMaSk/tree/master/DCIS/TrainData/overlay (Right)

### Illustrative training images for DCIS segmentation based on IM-NET

<div class="items">
<figure>
    <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_img_movie_001.gif" width="250" height="250">
    
</figure>
<figure>
    <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_mask_movie_001.gif" width="250" height="250"> 
    
</figure>
<figure>
    <img src="CIS/PrepareData/IM-NET/training_material/DCIS_freehand_sampled_pos_overlay_movie_001.gif" width="250" height="250"/>
    
 </figure>

</div>

## Citation

Priya Lakshmi Narayanan et al., (2021) Unmasking the immune microecology of ductal carcinoma in situ with deep learning. https://doi.org/10.1038/s41523-020-00205-5

## Data availability

All training data of carcinoma in situ regions that were annotated as a part of the project is made available in this github repository.
Training data tiles were anonymised from raw HE image tiles. Request for data access for the Duke samples can be submitted to E.S.H and Y.Y

### Notes

1. This project is a work in progress and contact the corresponding authors for any queries. Docker images are packaged in each of these modules are made available and few example test images are provided for users to start their experiment. All the modules are independant of each other so each module can be iteratively tested. Usage of tiled images will help to test the pipeline without performing tissue segmentation.



