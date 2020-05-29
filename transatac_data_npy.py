# -*- coding: utf-8 -*-
"""
Created on Wed May 23 07:16:11 2018

@author: priya.narayanan@icr.ac.uk
"""
##### Data preparation to be fed to the network #########
##### De-identify image names with the numerals #########

import os
import numpy as np
import glob
from skimage.io import imsave, imread
from skimage import io
import cv2



data_path = 'HE_Data\\'#folder name containing unet_traindata



image_rows = 512
image_cols = 512

def create_train_data():
    train_data_path = os.path.join(data_path, 'TA_Train_data')
    images = os.listdir(train_data_path)
    print (len(images))
    total = int(len(images) / 2)
    print (total)
    imgs = np.ndarray((total, image_rows, image_cols,3))
    imgs_mask = np.ndarray((total, image_rows, image_cols,1))

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        print("path=",os.path.join(train_data_path, image_name))        
        img = cv2.imread(os.path.join(train_data_path, image_name))

        
        img = cv2.resize(img,(int( 512),int(512)), interpolation = cv2.INTER_CUBIC)
        #print(img.shape)

        
        img_mask1 = cv2.imread(os.path.join(train_data_path, image_mask_name),0)
        img_mask1  = cv2.resize(img_mask1 ,(int( 512),int(512)), interpolation = cv2.INTER_CUBIC)
        #print(img_mask1.shape)
        img_mask1 = np.expand_dims(img_mask1,axis=-1)

        
        img = np.array([img])
        img_mask = np.array([img_mask1])
        
        imgs[i] = img
        imgs_mask[i] = img_mask



        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print(imgs[10].shape)
    print(imgs_mask[10].shape)
    #io.imshow(imgs_mask[10])
   
    np.save('he_imgs_train.npy', imgs)
    np.save('he_imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')
    
def load_train_data():
    imgs_train = np.load('he_imgs_train.npy')
    imgs_mask_train = np.load('he_imgs_mask_train.npy')
    return imgs_train.shape, imgs_mask_train.shape



if __name__ == '__main__':
    create_train_data()
    train,trainmask = load_train_data()
    print(train,trainmask)
     
    
