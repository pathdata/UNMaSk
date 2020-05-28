

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:15:26 2018

@author: pnarayanan
"""

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Add, AveragePooling2D
from keras.optimizers import Adam
from keras import backend as K

import cv2
import glob
import os
import numpy as np
from skimage.io import imsave, imread
import cv2
from PIL import Image
import re
from PIL import ImageEnhance

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
#%%
def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
#   """See http://www.codinghorror.com/blog/archives/001018.html""" 

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(weights_path=None):
    img_rows=512
    img_cols=512    
    inputs = Input((img_rows, img_cols,3))
#    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

#    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    if weights_path: 
        model.load_weights(weights_path) 
 
    return model


def inception_block(input_tensor, num_filters):

    p1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    p1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)

    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)

    p3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)

    # return concatenate([p1, p2, p3], axis=3)

    return Add()([p1, p2, p3])


def get_Inception_unet(weights_path=None):
    img_rows = 512
    img_cols = 512

    base = 16
    input_tensor = Input((img_rows, img_cols, 3))
    b1 = inception_block(input_tensor, base)

    pool1 = AveragePooling2D(pool_size=(2, 2))(b1)

    b2 = inception_block(pool1, base * 2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(b2)

    # b3  = self.inception_block(pool2, self.base*4)
    b3 = inception_block(pool2, base*4)
    pool3 = AveragePooling2D(pool_size=(2, 2))(b3)

    b4 = inception_block(pool3, base * 8)
    # b4 = self.inception_block(b4, self.base*4)

    up5 = concatenate([Conv2DTranspose(base * 4, (2, 2), strides=(2, 2), padding='same')(b4), b3], axis=3)
    b5  = inception_block(up5, base*4)
    # b5 = self.inception_block(b5, self.base*4)

    up6 = concatenate([Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding='same')(b5), b2], axis=3)
    b6 = inception_block(up6, base * 2)
    # b6 = self.inception_block(b6, self.base*2)

    up7 = concatenate([Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(b6), b1], axis=3)
    b7 = inception_block(up7, base)
    # b7 = self.inception_block(b7, self.base)

    b8 = Conv2D(1, (1, 1), activation='sigmoid')(b7)

    model = Model(inputs=[input_tensor], outputs=[b8])
    #model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    if weights_path:
        model.load_weights(weights_path)

    return model

    
#%%% 
model = get_Inception_unet('HE_100_weights.h5')

model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

#%%%    
def load_train_data():
    imgs_train = np.load('he_imgs_train.npy')
    imgs_mask_train = np.load('he_imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

#%%
def preprocess(imgs):
    img_rows=512
    img_cols=512
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows,3)) 

    return imgs_p
#%%
if __name__=="__main__":

    Path = r"HE_Data/TransATAC/TA_test"
    result_dir = r"HE_Data/TransATAC"

    if os.path.exists(os.path.join(result_dir, "probability_img")) is False:
        os.makedirs(os.path.join(result_dir, "probability_img"))

    if os.path.exists(os.path.join(result_dir, "pred_full_img")) is False:
        os.makedirs(os.path.join(result_dir, "pred_full_img"))

    if os.path.exists(os.path.join(result_dir, "predmask_full_img")) is False:
        os.makedirs(os.path.join(result_dir, "predmask_full_img"))


    names=os.listdir(Path)
    names = sorted(names, key=natural_key)
    blkSize = 512
    shiftSize = 256
    for im in names:
        img_name=im
        print(img_name)
        if (im[-3:]=="jpg"):
            #if (im[:2]=="10"):
            Img = cv2.imread(Path+"//"+im)
            nrow, ncol, nch = Img.shape
            blkSize = 512
            shiftSize = 256
            if((nrow%blkSize)!=0):
            #    newrow= int((math.ceil((nrow/blkSize))*blkSize)+ shiftSize)
                newrow= int(nrow+ shiftSize)
            else:
                newrow = int(nrow+shiftSize)
              
#                print(newrow)
            
            if((ncol%blkSize)!=0):
            #    newcol= int((math.ceil((ncol/blkSize))*blkSize)+ shiftSize)
                newcol= int(ncol+ shiftSize)
            else:
                newcol = int(ncol+shiftSize)
            

            
            new_Img = np.zeros((newrow, newcol, nch),np.uint8)
            out_Img = np.zeros((newrow, newcol),np.uint8)
            out_Img1 = np.zeros((newrow, newcol,3),np.uint8)                
            new_Img[0:nrow,0:ncol,0:3] = Img[0:nrow, 0:ncol,0:3]
            kernel = np.ones((5,5),np.float32)/25
            k=0
            tile_size=512                
            imgs = np.ndarray((1, tile_size, tile_size,3))
            i=0
            for r in range(0,(newrow-shiftSize),shiftSize):
                for c in range(0,(newcol-shiftSize),shiftSize):


                    blk = np.zeros((512, 512, 3), np.uint8)
                    blk = new_Img[r:(r+blkSize), c:(c+blkSize),0:3]

                    h1,w1,_=blk.shape

                    img1 = cv2.resize(blk,(int( tile_size),int(tile_size)), interpolation = cv2.INTER_CUBIC)

                    img_11=cv2.resize(img1,(w1,h1))
                    imgs[i] = img1

                    imgs_train, imgs_mask_train = load_train_data()
                           
                    imgs_train = preprocess(imgs_train)
                    imgs_mask_train = preprocess(imgs_mask_train)
                
                    imgs_train = imgs_train.astype('float32')
                    mean = np.mean(imgs_train)  # mean for data centering
                    std = np.std(imgs_train)  # std for data normalization
                    imgs_test_p=imgs
                    im= np.reshape(imgs_test_p, (512,512,3))

                    imgs_test_p = preprocess(imgs_test_p)
                
                    imgs_test_p = imgs_test_p.astype('float32')
                    imgs_test_p -= mean
                    imgs_test_p /= std
            
            
                    imgs_mask_test_p = model.predict(imgs_test_p, verbose=1)


                    imgs_mask_test_p= np.reshape(imgs_mask_test_p, (512,512,1))
                    imgs_mask_test_p=cv2.resize(imgs_mask_test_p,(512,512))
                    imgs_mask_test_p=imgs_mask_test_p * 255 

                    out_1=np.zeros(((512-8), (512-8),3),np.uint8)
                    h_1=512-8
                    w_1=512-8
                    r_1=4
                    c_1=4
                    

                    out_1=imgs_mask_test_p[r_1:(r_1+h_1), c_1:(c_1+w_1)]

                    imgs_mask_test_p1 =out_1
                    imgs_mask_test_p1=cv2.resize(imgs_mask_test_p1,(w1,h1)) 
                    
                    out_Img[r:(r+blkSize), c:(c+blkSize)] =imgs_mask_test_p1[0:blkSize,0:blkSize]

                    k=k+1
                        
                finalImg = np.zeros((nrow, ncol), np.uint8)
                finalImg[0:nrow,0:ncol] = out_Img[0:nrow, 0:ncol]

                
                prob_image=finalImg
                w,h,=prob_image.shape
                test_image=Img
                w1,h1,_=test_image.shape
                #prob_image = cv2.cvtColor(prob_image,cv2.COLOR_BGR2GRAY)

                prob_image_smooth = cv2.GaussianBlur(prob_image,(251,251),0)
                  
                gray=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) 
               
                yImage3=np.zeros([h,w])
                yImage3=yImage3.astype('uint8')
                yImage3=cv2.resize(prob_image_smooth,yImage3.shape,interpolation=cv2.INTER_LINEAR)

                yImage3 = cv2.applyColorMap(yImage3, cv2.COLORMAP_JET)
                      
                dst=cv2.addWeighted(yImage3,0.5,test_image,0.5,0)
                cv2.imwrite(os.path.join(result_dir,"pred_full_img",img_name),finalImg.astype(np.uint8))
                cv2.imwrite(os.path.join(result_dir,"probability_img",img_name) ,dst)

                
                
                               
                                        

                                         