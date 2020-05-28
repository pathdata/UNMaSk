
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Add, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

#from poet_data_npy import load_train_data
#from unet_train_data import load_test_data
#from data import *
import matplotlib.pyplot as plt


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#%%
img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def load_train_data():
    imgs_train = np.load('he_imgs_train.npy')
    imgs_mask_train = np.load('he_imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def get_unet():
    
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

    #model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def inception_block(input_tensor, num_filters):

    p1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    p1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)

    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)

    p3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)

    # return concatenate([p1, p2, p3], axis=3)

    return Add()([p1, p2, p3])


def get_Inception_unet():

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
    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    return model




def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows,3)) 

    return imgs_p


def preprocess_mask(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,1), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows,1))

    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_train_data()

    print("img_train=",imgs_train.shape)
    print("img_mask_train=",imgs_mask_train.shape)
    
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess_mask(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_Inception_unet()
    model_checkpoint = ModelCheckpoint('HE_100_weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    print("img_train=",imgs_train.shape)
    print("img_mask_train=",imgs_mask_train.shape)
    H=model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=100, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    print("finished...")
    return H


if __name__ == '__main__':

    # M = get_Inception_unet()
    # print(M.summary())



    H = train_and_predict()
    plt.style.use("seaborn-white")
    plt.figure()
    plt.plot(np.arange(0,100),H.history["loss"],label="train_loss")
    plt.plot(np.arange(0,100),H.history["val_loss"],label="val_loss")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('HE_100_Loss.png')

    plt.figure()
    plt.plot(np.arange(0,100),H.history["dice_coef"],label="dice_coef")
    plt.plot(np.arange(0,100),H.history["val_dice_coef"],label="val_dice_coef")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('HE_100_Accuracy.png')






