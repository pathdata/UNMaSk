import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)    #1.10.0
import keras
print(keras.__version__) #2.2.4
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D, AveragePooling2D, Add
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras import backend as K
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def plot_training_curves(output_dir, H, epochs):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use("seaborn-white")
    plt.figure()
    plt.subplot(211)
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


    plt.subplot(212)
    plt.plot(H.history["acc"], label="train_accuracy")
    plt.plot(H.history["val_acc"], label="val_accuracy")
    # plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, str(epochs) + 'HE_100_BU_Accuracy.png'))

def inception_block(input_tensor, num_filters):

    p1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    p1 = BatchNormalization()(p1)
    p1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)

    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    p2 = BatchNormalization()(p2)
    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)

    p3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)

    # return concatenate([p1, p2, p3], axis=3)

    return Add()([p1, p2, p3])


def get_Inception_unet():

    base = 16
    img_rows = 512
    img_cols = 512
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

def train_and_predict(batch_size, epochs):

    save_dir = './model_HE_Inception_unet'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)


    X_train = np.load('Train_HE/he_imgs_train.npy')
    X_train = X_train.astype('float32')
    y_train = np.load('Train_HE/he_imgs_mask_train.npy')

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]

    X_valid = np.load('Train_HE/imgs_valid.npy')
    X_valid = X_valid.astype('float32')
    y_valid = np.load('Train_HE/imgs_mask_valid.npy')
    y_valid = y_valid.astype('float32')
    y_valid /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = get_Inception_unet()
    model.summary()
    model.compile(optimizer=Adam(), loss=["binary_crossentropy"], metrics=["accuracy"])
    callbacks = [
        EarlyStopping(patience=40, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(os.path.join(save_dir, 'model-tissue-seg.h5'), verbose=1, save_best_only=True,
                        save_weights_only=True)
    ]

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    print('-' * 30)
    print('Training data description...')
    print('-' * 30)

    print("img_train=", X_train.shape)
    print("img_mask_train=", y_train.shape)

    print('-' * 30)
    print('validation data description...')
    print('-' * 30)

    print("img_valid=", X_valid.shape)
    print("img_mask_valid=", y_valid.shape)


    H = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                  validation_data=(X_valid, y_valid))

    print("finished...")

    return H

if __name__=="__main__":

    params = {'batch_size': 4,
              'epochs': 100
              }

    epochs = params['epochs']

    H = train_and_predict(**params)
    plot_training_curves(output_dir='./history', H=H, epochs=params['epochs'])


