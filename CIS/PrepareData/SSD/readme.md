Understanding Data preparation 

In this section we will summarize the organization of directory struture to enable the end user extract the information needed to directly train the object detection models.
CIS-|_
       PrepareData-|_
                     SSD-|_
                           square_raw-|_
                                  img1.svs
                                  img1.xml
                                  img2.svs
                                  img2.xml
                                  T1.ndpi
                                  T1.xml
                            square_annotation-|_
                                      img_level2_5x-|_
                                            pos-|_
                                                T1XXXX.jpg
                                                T1XXXX.xml
                                            neg-|_
                                                T2XXXX.jpg
                                            rect-|_
                                                 T1XXXX.jpg
Square raw directory contains the whole slide image and the respective annotations.
Square annotation directory contains positive example, negative example and the rectangle annotations overlayed on the image.
