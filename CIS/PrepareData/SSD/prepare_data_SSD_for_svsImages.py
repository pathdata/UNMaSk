# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:49:03 2018

@author: pnarayanan
"""

import openslide
import os

import cv2
import xml.dom.minidom
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element,ElementTree
import numpy as np
from PIL import Image

from prt import prettify

################ Data Preparation for SSD ############################


class GenerateWSI_squareannotation_on_cws(object):

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

        file_names_list = [fname for fname in os.listdir(self.input_slide_dir) if fname.endswith(self.ext) is True]


        if os.path.exists(os.path.join(self.output_dir,'img_level2_5x')) is False:
            os.makedirs(os.path.join(self.output_dir,'img_level2_5x'))

        if os.path.exists(os.path.join(self.output_dir,'img_level2_5x','pos')) is False:
            os.makedirs(os.path.join(self.output_dir,'img_level2_5x','pos'))

        if os.path.exists(os.path.join(self.output_dir,'img_level2_5x','neg')) is False:
            os.makedirs(os.path.join(self.output_dir,'img_level2_5x','neg'))

        if os.path.exists(os.path.join(self.output_dir,'img_level2_5x','rect')) is False:
            os.makedirs(os.path.join(self.output_dir,'img_level2_5x','rect'))


        #ds=int(osr.level_downsamples[level])
        #w,h=osr.level_dimensions[0]
        #%%
        #os.path.join(self.input_slide_dir,os.path.splitext(slide)[0]+'.xml')

        #slide_im=os.listdir("square_raw_DUKE")
        for slide in file_names_list:

                doc = xml.dom.minidom.parse(os.path.join(self.input_slide_dir,os.path.splitext(slide)[0]+'.xml'))
                Region = doc.getElementsByTagName("Region")

                X=[]
                Y=[]
                label=[]
                for Reg in Region:
                    directory = Reg.getAttribute("Text")
                    if (directory==u''):
                        label.append("none")
                    else:
                        label.append(directory)
                    Vertex = Reg.getElementsByTagName("Vertex")
                    for Vert in Vertex:
                            X.append(Vert.getAttribute("X"))
                            Y.append(Vert.getAttribute("Y"))
                x=[]
                y=[]

                for i in range(0,len(X),2):

                    x.append(int(float(X[i])))

                for i in range(0,len(Y),2):

                    y.append(int(float(Y[i])))

                for i in range(0,len(x),2):
                  if(x[i+1]<x[i]):
                      x[i],x[i+1]=x[i+1],x[i]

                for i in range(0,len(y),2):
                  if(y[i+1]<y[i]):
                      y[i],y[i+1]=y[i+1],y[i]

                #Added Extraction of annotation in different levels from the annotation region

                lvl=1



                level=lvl
                osr=openslide.OpenSlide(os.path.join(self.input_slide_dir,slide))
                ds=int(osr.level_downsamples[level])
                w,h=osr.level_dimensions[0]

                k=0

                for j in range(0,h,2000):   #i swap this two lines
                    for i in range(0,w,2000):

                        if(i+2000>w):
                            width=w-i
                        else:
                            width=2000
                        if(j+2000>h):
                            height=h-j
                        else:
                            height=2000
                        width=int(width/ds)
                        height=int(height/ds)
                        out=osr.read_region((i,j),level,(width,height))
                        temp = np.array(out)
                        temp = temp[:, :, 0:3]
                        out = Image.fromarray(temp)
                        out.save("img.jpg")
                        img=cv2.imread("img.jpg")
                        img_annotation_test=img
                        flag=0
                        label_ind=0
                        for i1 in range(0,len(x),2):
                            if i<x[i1+1] and i+2000>x[i1] and j<y[i1+1] and j+2000>y[i1]:
                                if (flag==0):
                                    flag=1

                                    wi, he = out.size




                                    root = Element("annotation")
                                    tree = ElementTree(root)

                                    folder=Element("folder")
                                    folder.text = 'images'
                                    root.append(folder)

                                    filename=Element("filename")
                                    filename.text = os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+".jpg"
                                    root.append(filename)

                                    path=Element("filename")
                                    current_path=os.getcwd()
                                    path.text = current_path+"\\img\\"+os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+".jpg"
                                    root.append(path)

                                    source=Element("source")
                                    root.append(source)
                                    database=ET.SubElement(source,"database")
                                    database.text='Unknown'

                                    size=Element("size")
                                    root.append(size)
                                    width=ET.SubElement(size,"width")
                                    width.text=str(wi)
                                    height=ET.SubElement(size,"height")
                                    height.text=str(he)
                                    depth=ET.SubElement(size,"depth")
                                    depth.text='3'
                                    segmented=Element("segmented")
                                    segmented.text = '0'
                                    root.append(segmented)





                                #print(width, height)
                                print((x[i1]-i)/ds,(y[i1]-j)/ds,(x[i1+1]-i)/ds,(y[i1+1]-j)/ds)
                                left=int((x[i1]-i)/ds)
                                if(left<0):
                                    left=0
                                top=int((y[i1]-j)/ds)
                                if (top<0):
                                    top=0
                                right=int((x[i1+1]-i)/ds)
                                if (right>wi):
                                    right=wi
                                bottom=int((y[i1+1]-j)/ds)
                                if(bottom>he):
                                    bottom=he

                                Object=Element("object")
                                root.append(Object)
                                name=ET.SubElement(Object,"name")
                                name.text=label[label_ind]
                                pose=ET.SubElement(Object,"pose")
                                pose.text='Unspecified'
                                truncated=ET.SubElement(Object,"truncated")
                                truncated.text='0'
                                difficult=ET.SubElement(Object,"difficult")
                                difficult.text='0'
                                bndbox=ET.SubElement(Object,"bndbox")
                                xmin=ET.SubElement(bndbox,"xmin")
                                xmin.text=str(left)
                                ymin=ET.SubElement(bndbox,"ymin")
                                ymin.text=str(top)
                                xmax=ET.SubElement(bndbox,"xmax")
                                xmax.text=str(right)
                                ymax=ET.SubElement(bndbox,"ymax")
                                ymax.text=str(bottom)
                                label_ind+=1
                                cv2.rectangle(img_annotation_test,(left,top),(right,bottom),(0,255,0),3)
                                cv2.imwrite(os.path.join(self.output_dir,'img_level2_5x','rect',os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+".jpg"),img_annotation_test)
                        if(flag==1):
                                cv2.imwrite(os.path.join(self.output_dir,"img_level2_5x","pos",os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+".jpg"),img)
                                tree.write(os.path.join(self.output_dir,"img_level2_5x","pos",os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+'.xml'))
                                with open(os.path.join(self.output_dir,"img_level2_5x","pos",os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+'.xml'), 'w') as f:
                                     f.write(prettify(root))
                        else:
                                 cv2.imwrite(os.path.join(self.output_dir,"img_level2_5x","neg",os.path.splitext(slide)[0]+'Da'+'_'+'{0:04}'.format(k)+"_"+str(level)+".jpg"),img)
                        k+=1

if __name__ == '__main__':
    params  = {'input_slide_dir': r'D:/TF_TISSUE/HE_Tissue_seg/square_raw_DUKE',             # input slide dir
               'output_dir': r'D:/TF_TISSUE/HE_Tissue_seg/square_annotation',                # output dir 1.img_mask,2.Mat_files
               'ext':'.svs',                                                                 # ext: '.svs'
               }
    obj = GenerateWSI_squareannotation_on_cws(**params)
    obj.generate_patch_of_annotated_tiles()
