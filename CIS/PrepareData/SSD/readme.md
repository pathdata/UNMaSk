###Understanding Data preparation for object detection models 

In this section we will summarize the organization of directory struture to enable the end user extract the information needed to directly train the object detection models.

CIS
 |
 |
 +-- PrepareData
 |    
 |
 |  |  
 |  \-- SSD
 |       +-- square_raw
 |              +-----img1.svs
 |                    img1.xml
 |                    img2.svs
 |                    img2.xml
 |                    T1.ndpi
 |                    T1.xml
 |       +-- square_annotation
 |               +-----img_level2_5x
 |                         +-------
 |                               pos
 |                                 +-
 |                                  T1XXXX.jpg
 |                                  T1XXXX.xml
 |                               neg
 |                                  +-
 |                                   T2XXXX.jpg
 |                               rect
 |                                   +-
 |                                    T1XXXX.jpg
 |    
 |
 |   
 |  
 |   
 |  \-- IM-NET
 |    
 +-- dir 4


                            
Square raw directory contains the whole slide image and the respective annotations.
Square annotation directory contains positive example, negative example and the rectangle annotations overlayed on the image.
