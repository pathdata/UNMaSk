# -*- coding: utf-8 -*-
"""
Created on Wed May 23 06:52:15 2018

@author: priya.narayanan
"""

import cv2
import os
import glob
import re

def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
#   """See http://www.codinghorror.com/blog/archives/001018.html"""                  

data_path="HE_Data"
immg="TA_Train_data"


if os.path.exists(os.path.join(data_path,immg)) is False:
    os.makedirs(os.path.join(data_path,immg))


list2=os.listdir(data_path+"\\TransATAC\\TA_mask")
list2 = sorted(list2,key=natural_key)
n=0
for list12 in list2:
    print(list12)
    img=cv2.imread(data_path+"\\TransATAC\\TA_mask\\"+list12)
    cv2.imwrite(data_path+"\\TA_Train_data\\"+str(n)+"_mask.jpg",img)
    n=n+1
m=0        
list2=os.listdir(data_path+"\\TransATAC\\TA_orig")
list2 = sorted(list2,key=natural_key)
for list12 in list2:
    print(list12)
    img=cv2.imread(data_path+"\\TransATAC\\TA_orig\\"+list12)
    cv2.imwrite(data_path+"\\TA_Train_data\\"+str(m)+".jpg",img)
    m=m+1    