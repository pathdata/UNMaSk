# Understanding Data preparation for object detection models 

In this section we will summarize the organization of directory struture to enable the end user extract the information needed to directly train the object detection models.

1. parseCIS

<p align="center">
  <img src="training_material/tree_structure.png" width="350"/>
  <img src="training_material/train_square_demo.png" width="350"/>
</p>

 
                            
Square raw directory contains the whole slide image and the respective annotations.
Square annotation directory contains 1. positive example 2. negative example and 3.rectangle annotations overlayed on the image.
