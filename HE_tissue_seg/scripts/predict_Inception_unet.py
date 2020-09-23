
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, BatchNormalization, AveragePooling2D, Add

from keras import backend as K

import cv2

import re

from scripts.slidingpatches import *


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
#%%
def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



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
    b3 = inception_block(pool2, base * 4)
    pool3 = AveragePooling2D(pool_size=(2, 2))(b3)

    b4 = inception_block(pool3, base * 8)
    # b4 = self.inception_block(b4, self.base*4)

    up5 = concatenate([Conv2DTranspose(base * 4, (2, 2), strides=(2, 2), padding='same')(b4), b3], axis=3)
    b5 = inception_block(up5, base * 4)
    # b5 = self.inception_block(b5, self.base*4)

    up6 = concatenate([Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding='same')(b5), b2], axis=3)
    b6 = inception_block(up6, base * 2)
    # b6 = self.inception_block(b6, self.base*2)

    up7 = concatenate([Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(b6), b1], axis=3)
    b7 = inception_block(up7, base)
    # b7 = self.inception_block(b7, self.base)

    b8 = Conv2D(1, (1, 1), activation='sigmoid')(b7)

    model = Model(inputs=[input_tensor], outputs=[b8])
    # model.compile(optimizer=Adam(lr=1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    if weights_path:
        model.load_weights(weights_path)


    return model


# %%%


if __name__=="__main__":

    Path = r"test\HE"
    result_dir = r"test\test_IUB"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    if os.path.exists(os.path.join(result_dir, "probability_img")) is False:
        os.makedirs(os.path.join(result_dir, "probability_img"))

    if os.path.exists(os.path.join(result_dir, "pred_full_img")) is False:
        os.makedirs(os.path.join(result_dir, "pred_full_img"))

    if os.path.exists(os.path.join(result_dir, "predmask_full_img")) is False:
        os.makedirs(os.path.join(result_dir, "predmask_full_img"))

    model = get_Inception_unet('model_HE_Inception_unet/model-tissue-seg.h5')
    model.summary()

    for img in os.listdir(Path):

        if img.endswith('.jpg') is True:


            img_name =img

            image = Image.open(os.path.join(Path, img))
            image_np = np.array(image)

            patch_obj = Slidingpatches(
                img_patch_h=512, img_patch_w=512,
                stride_h=256, stride_w=256,
                label_patch_h=512, label_patch_w=512)

            data = patch_obj.extract_patches(image_np)



            print(data.shape)

            y_pred = []


            for i in range(0, data.shape[0]):

                p_img = data[i]
                p_img = np.expand_dims(p_img, axis=0)

                #imgs_mask_test_p = model.predict(p_img, verbose=1)
                imgs_mask_test_p = model.predict(p_img)

                #print(imgs_mask_test_p.shape)

                pred = np.squeeze(imgs_mask_test_p)

                #print(pred.shape)
                y_pred.append(pred)
            y_pred = np.array(y_pred)
            output = patch_obj.merge_patches(y_pred)
            label = (255 * output).astype('uint8')
            labelt = label > 120
            labelt = (labelt*255).astype('uint8')
            cv2.imwrite(os.path.join(result_dir, "predmask_full_img", img_name), labelt)

            prob_image = labelt
            w, h, = prob_image.shape

            prob_image_smooth = cv2.GaussianBlur(prob_image, (51, 51), 0)

            yImage3 = np.zeros([h, w])
            yImage3 = yImage3.astype('uint8')
            yImage3 = cv2.resize(prob_image_smooth, yImage3.shape, interpolation=cv2.INTER_LINEAR)

            yImage3 = cv2.applyColorMap(yImage3, cv2.COLORMAP_JET)


            dst = cv2.addWeighted(yImage3, 0.5, image_np, 0.5, 0)

            cv2.imwrite(os.path.join(result_dir, "pred_full_img", img_name), prob_image_smooth)
            cv2.imwrite(os.path.join(result_dir, "probability_img", img_name), dst)
        else:
            continue



        print(output.shape)
