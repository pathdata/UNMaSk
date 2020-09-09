# Understanding data preparation for object detection models 

In this section we will summarize the organization of directory struture to enable the end user extract the information needed to directly train the object detection models.

1. parseCIS

<p align="center">
  <img src="training_material/tree_structure.png" width="350"/>
  <img src="training_material/tree_svs_demo.PNG" width="350"/>
</p>

<p align="center">
  <img src="training_material/train_square_demo.png" width="350"/>
  <img src="training_material/train_svs_demo.png" width="350"/>
</p>
 
                            
Square raw directory contains the whole slide images and the respective annotations.
Square annotation directory contains 
    1. positive example 
    2. negative example 
    3.rectangle annotations overlayed on the image for visualisation.

# Visualisation of the square annotations as regions of individual tiles from the whole slide images
<p align="center">
  <img src="training_material/test_square_annotaions_DCIS_movie.gif" width="350" height="350"/>
</p>

