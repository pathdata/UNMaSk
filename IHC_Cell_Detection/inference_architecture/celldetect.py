
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Add, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


import tensorflow as tf
import matplotlib.pyplot as plt


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# %%
img_rows = 224
img_cols = 224

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1- dice_coef(y_true, y_pred)
	
def ModHausdorffDist(y_true,y_pred):
    y_true_f = tf.reshape(y_true,(2,-1))#1d
    y_pred_f = tf.reshape(y_pred,(2,-1))#1d
    fd = K.mean(K.min(K.sqrt(tf.concat(tf.dtypes.cast(K.sum(y_true_f * y_true_f, axis = 1) + K.sum(y_pred_f * y_pred_f, axis = 1) -2 * K.sum(y_true_f *  y_pred_f, axis = 1),tf.float32),axis=0))))
    return fd


def load_train_data():
    imgs_train = np.load('npy_files/train_images_aug.npy')
    imgs_mask_train = np.load('npy_files/train_labels_aug.npy')
    imgs_mask_train = (imgs_mask_train > 200) * 1
    return imgs_train, imgs_mask_train


def load_validation_data():
    imgs_valid = np.load('npy_files/val_images.npy')
    imgs_mask_valid = np.load('npy_files/val_labels.npy')
    imgs_mask_valid = (imgs_mask_valid > 200) * 1
    return imgs_valid, imgs_mask_valid


def plot_training_curves(output_dir, H):
    plt.style.use("seaborn-white")
    plt.figure()
    plt.plot(np.arange(0, 1000), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 1000), H.history["val_loss"], label="val_loss")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'A1000_3INS_Loss.png'), dpi=800)
    plt.show()

    plt.figure()
    plt.plot(np.arange(0, 1000), H.history["dice_coef"], label="dice_coef")
    plt.plot(np.arange(0, 1000), H.history["val_dice_coef"], label="val_dice_coef")
    plt.title("Training/Validation Networrk")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'A1000_3INS_Accuracy.png'), dpi=800)
    plt.show()


def get_unet3():
    inputs = Input((img_rows, img_cols, 3))
    base = 16
    conv1 = Conv2D(base, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(base, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(base * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(base * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(base * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(base * 4, (3, 3), activation='relu', padding='same')(conv3)

    up8 = concatenate([Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv8 = Conv2D(base * 2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(base * 2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(base, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(base, (3, 3), activation='relu', padding='same')(conv9)

    unet_output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[unet_output])
    #model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, 'accuracy'])
    model.compile(optimizer=Adam(lr=0.0001), loss=[ModHausdorffDist], metrics=[dice_coef,'accuracy'])


    return model

def inception_block(input_tensor, num_filters):

    p1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    p1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)

    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    p2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)

    p3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)

    # return concatenate([p1, p2, p3], axis=3)

    return Add()([p1, p2, p3])

#=========================================================================
#Concatenate requires the two tensors to have the same shape except the concatenate axis
#Method_1: crop the layer to the shape of transpose layer
#Ref https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

#Method_2: reshape the other axises to match the transpose layer
def reshape_axis(target,refer):
    a0,a1,a2 = refer.shape[0],refer.shape[1],refer.shape[2]
    target = tf.reshape(target,(a0,a1,a2,-1))
    return(K.cast(target,"float32"))
#Method_3: add pooling layer
#========================================================================

def get_Inception_unet(weights_path=None):

    base = 16
    input_tensor = Input((img_rows, img_cols, 3))
    b1 = inception_block(input_tensor, base)

    pool1 = AveragePooling2D(pool_size=(2, 2))(b1)

    b2 = inception_block(pool1, base * 2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(b2)

    b3 = inception_block(pool2, base*4)
    pool3 = AveragePooling2D(pool_size=(2, 2))(b3)

    b4 = inception_block(pool3, base * 8)
    
    up5 = concatenate([Conv2DTranspose(base * 4, (2, 2), strides=(2, 2), padding='same')(b4), b3], axis=3)

    print(up5.shape, b1.shape)
    
    up5 = concatenate([up5,AveragePooling2D(pool_size=(4,4))(b1)], axis = 3)

    b5  = inception_block(up5, base*4)
    # b5 = self.inception_block(b5, self.base*4)

    up6 = concatenate([Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding='same')(b5), b2], axis=3)

    print(up6.shape, b3.shape)
    
    up_b3 = Conv2DTranspose(base*2,(2,2),strides=(2,2),padding='same')(b3)
    up6 = concatenate([up6,up_b3], axis = 3)

    b6 = inception_block(up6, base * 2)
    # b6 = self.inception_block(b6, self.base*2)

    up7 = concatenate([Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(b6), b1], axis=3)
    b7 = inception_block(up7, base)
    # b7 = self.inception_block(b7, self.base)

    b8 = Conv2D(1, (1, 1), activation='sigmoid')(b7)

    model = Model(inputs=[input_tensor], outputs=[b8])
    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss], metrics=["accuracy", dice_coef])

    if weights_path:
        model.load_weights(weights_path)

    return model






def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows, 3))
    #        imgs_p[i] = resize(imgs[i], (img_cols, img_rows,3), preserve_range=True)

    #    imgs_p = imgs_p[..., np.newaxis]
    #    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def preprocess_mask(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows, 1))
    #        imgs_p[i] = resize(imgs[i], (img_cols, img_rows,3), preserve_range=True)

    #    imgs_p = imgs_p[..., np.newaxis]
    #    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()
    print("img_train=", imgs_train.shape)
    print("img_mask_train=", imgs_mask_train.shape)

    img_train_test = imgs_train[0]
    print(np.max(img_train_test[:, :, 0]))
    print(np.min(img_train_test[:, :, 0]))

    #
    imgs_train = imgs_train * 1.0 / 255
    imgs_train = imgs_train.astype('float32')
    #
    imgs_mask_train = imgs_mask_train.astype('float32')
    # imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_valid, imgs_mask_valid = load_validation_data()
    print("imgs_valid=", imgs_valid.shape)
    print("imgs_mask_valid=", imgs_mask_valid.shape)

    imgs_valid = imgs_valid * 1.0 / 255
    imgs_valid = imgs_valid.astype('float32')
    #
    imgs_mask_valid = imgs_mask_valid.astype('float32')

    #
    print('-' * 50)
    print('Creating and compiling model...')
    print('-' * 50)
    model = get_Inception_unet()
    #model.summary()
    model_checkpoint = ModelCheckpoint('model/A1000_3INS_Aug_Lr_0.0001_Dice_weights.h5', monitor='val_loss', save_best_only=True)

    print('-' * 50)
    print('Fitting model...')
    print('-' * 50)

    print("img_train=", imgs_train.shape)
    print("img_mask_train=", imgs_mask_train.shape)
    H = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=1000, verbose=1, shuffle=True,
                  validation_data=(imgs_valid, imgs_mask_valid),
                  callbacks=[model_checkpoint])
    print("finished...")

    return H


if __name__ == '__main__':
    H = train_and_predict()
    plot_training_curves(output_dir='history', H=H)








