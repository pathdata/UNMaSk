# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:58:01 2018

@author: priya.narayanan@icr.ac.uk
"""
#
#This code will convert the aperio free hand drawing and save the binary masks.
# cws tiles of size 2000x2000 will be saved in img directory
# binary masks will be saved in the output_dir/slidename/img_mask directory.

##level =0 specifies the base magnification##
##To test on a smaller magnification change the level =3 to test for .ndpi images and level = 2 for svs images ##
# This saves the input cws tiles and the masks in the out directory in lower magnification.



#TO DO We need to make a outer contour of the binary masks.

import xml.dom.minidom
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element,ElementTree
from PIL import Image
import numpy as np
import cv2
import re
import openslide
import os
import glob
import scipy.io as iosci
from skimage import io
from patches import *
from slidingpatches import *

def natural_key(string_):
   
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


class GenerateWSIannotation_on_cws(object):

    def __init__(self,
                 input_slide_dir,
                 output_dir,
                 ext):
        self.input_slide_dir = input_slide_dir
        self.output_dir = output_dir
        self.ext = ext
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)


    def generate_patch_of_annotated_tiles(self):

        # Annotated tiles positive images will be saved in Mat_files/pos directory
        # Background tiles will be saved in Mat_files directory
        # .mat files from this will be used to generate tf records file

        file_names_list = [fname for fname in os.listdir(self.input_slide_dir) if fname.endswith(self.ext) is True]

        cws_dir = "cws"

        if os.path.exists(os.path.join(self.output_dir, cws_dir)) is False:
            os.makedirs(os.path.join(self.output_dir, cws_dir))

        for slide in file_names_list:

            if os.path.exists(os.path.join(self.output_dir, cws_dir, slide)) is False:
                os.makedirs(os.path.join(self.output_dir, cws_dir, slide))

            if os.path.exists(os.path.join(self.output_dir, cws_dir, slide, "img_mask")) is False:
                os.makedirs(os.path.join(self.output_dir, cws_dir, slide, "img_mask"))


            if os.path.exists(os.path.join(self.output_dir, cws_dir, slide, "Mat_files")) is False:
                os.makedirs(os.path.join(self.output_dir, cws_dir, slide, "Mat_files"))


            osr=openslide.OpenSlide(os.path.join(self.input_slide_dir,slide))
            level=0
            ds=osr.level_downsamples[level]
            w,h=osr.level_dimensions[0]

            ############################################################
            #Uncomment this section if you have not created the cws/image
            #tiles of size 2000x2000 from the raw whole slide images.
            ############################################################

            width=2000
            height=2000

            k=0
            for j in range(0,h,2000):
                for i in range(0,w,2000):

                    if(i+2000>w):
                        width=w-i
                    else:
                        width=2000
                    if(j+2000>h):
                        height=h-j
                    else:
                        height=2000
                    height=int(height/ds)
                    width=int(width/ds)
                    out=osr.read_region((i,j),level,(width,height))
                    temp = np.array(out)
                    temp = temp[:, :, 0:3]
                    out = Image.fromarray(temp)
                    out.save(os.path.join(self.output_dir, cws_dir,slide +'/Da'+str(k)+".jpg"))
                    k+=1

            mask_path = os.path.join(self.output_dir, cws_dir, slide, "img_mask")


            doc = xml.dom.minidom.parse(os.path.join(self.input_slide_dir,os.path.splitext(slide)[0]+'.xml'))
            Region = doc.getElementsByTagName("Region")

            X=[]
            Y=[]
            i=0
            for Reg in Region :
                X.append([])
                Y.append([])
                Vertex = Reg.getElementsByTagName("Vertex")
                for Vert in Vertex:

                    X[i].append(int(round(float(Vert.getAttribute("X")))))
                    Y[i].append(int(round(float(Vert.getAttribute("Y")))))
                i+=1



            i1=0
            points=[]
            for j in range(0,h,2000):
                for i in range(0,w,2000):
                   img=io.imread(os.path.join(self.output_dir, cws_dir,slide,'Da'+str(i1)+".jpg"))
                   [hh,ww,cc]=img.shape
                   blank_image = np.zeros((hh,ww), np.uint8)
                   for k in range(len(X)):

                        #print("######")
                        if i<max(X[k]) and i+2000>min(X[k]) and j<max(Y[k]) and j+2000>min(Y[k]):

                            points=[]
                            for i3 in range(len(X[k])):
                                points.append([int((X[k][i3]-i)/ds),int((Y[k][i3]-j)/ds)])
                            pts = np.array(points, np.int32)
                            pts = pts.reshape((-1,1,2))

                            cv2.drawContours(blank_image,[pts],0,(255),-1)
                   cv2.imwrite(os.path.join(mask_path ,'Da'+str(i1)+".jpg"),blank_image)

                   i1+=1



            mat_path = os.path.join(self.output_dir, cws_dir, slide, "Mat_files")

            positive_patch_path = os.path.join(mat_path, "pos")

            if os.path.exists(positive_patch_path) is False:
                os.makedirs(positive_patch_path)



            list_img= os.listdir(mask_path)
            list_img = sorted(list_img,key=natural_key)

            for list11 in list_img:

                gray=Image.open(os.path.join(mask_path,list11))
                tempgray = np.array(gray)
                gray1 = np.where(tempgray==255)
                temp = np.array(gray)
                y = np.expand_dims(temp, axis=-1)
######################### create sliding patches using Patches and Slidingpatches ####################################################
                if len(gray1[0])!=0:

                    img=Image.open(os.path.join(self.output_dir, cws_dir, slide,list11))
                    tempimg = np.array(img)


                    mask_patch_obj = Patches(
                        img_patch_h=600, img_patch_w=600,
                        stride_h=400, stride_w=400,
                        label_patch_h=600, label_patch_w=600)

                    img_patch_obj = Slidingpatches(
                        img_patch_h=600, img_patch_w=600,
                        stride_h=400, stride_w=400,
                        label_patch_h=600, label_patch_w=600)

                    train_patch = mask_patch_obj.extract_patches(y)

                    train_img_patch = img_patch_obj.extract_patches(tempimg)

###########################Sanity check for 2000x2000 tile size, without any stride 25 patches will be generated##################################

                    print(train_patch.shape[0])

                    #print(train_img_patch.shape[0])

                    for i in range(0, train_patch.shape[0]):

                        #sanity = np.where(np.squeeze(train_patch[i])) == 255
                        img_bw = cv2.findNonZero(np.squeeze(train_patch[i]))

                        if not img_bw is None:

                            data = {}
                            data['im'] = train_img_patch[i]
                            data['Mask'] = np.squeeze(train_patch[i])
                            GT = {}
                            GT['GT'] = data


                            iosci.savemat(os.path.join(positive_patch_path, list11[:-4] + "_" + str(i) + "_" + os.path.splitext(slide)[0] + ".mat"), GT)
                        else:
                            data = {}
                            data['im'] = train_img_patch[i]
                            data['Mask'] = np.squeeze(train_patch[i])
                            GT = {}
                            GT['GT'] = data

                            iosci.savemat(os.path.join(mat_path, list11[:-4] + "_" + str(i) + "_" + os.path.splitext(slide)[0] + ".mat"), GT)



                    # im1=img
                    # Mask=train_patch[1]
                    # data = {}
                    # data['im'] = im1
                    # data['Mask'] = Mask
                    # GT  = {}
                    # GT['GT']=data
                    # iosci.savemat(os.path.join(mat_path,list11[:-4]+"_"+os.path.splitext(slide)[0]+".mat"),GT)

# Data preparation by extracting the annotations at the base magnification.
# This code can be adapted to extract annotations at multiple resolutions.
# input_slide_dir : Path for the slide images
# output dir : Path containing mask and mat files. The mat files are part of data preparation for IM-Net and MicroNet
# training.

# Please note the comments in the the commented section above if the cws is not created already by running generatecws.py script
# This script can be modified to read the param.p files and then use it with the slight modification ###############

# Publication title : Unmasking the immune microecology of ductal carcinoma in situ with deep learning ###
# Author : priya.narayanan@icr.ac.uk
# Latest modified Date : 01-May-2020
if __name__ == '__main__':
    params  = {'input_slide_dir': r'D:\TF_TISSUE\HE_Tissue_seg\testsvs',             # input slide dir
               'output_dir': r'D:\TF_TISSUE\HE_Tissue_seg\svs_annotations_20X',       # output dir 1.img_mask,2.Mat_files
               'ext':'.ndpi',
               }

    obj = GenerateWSIannotation_on_cws(**params)
    obj.generate_patch_of_annotated_tiles()

                
