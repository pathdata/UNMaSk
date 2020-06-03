import tensorflow as tf
import scipy.io as sio
import os
import random
import pathlib
import numpy as np
from PIL import Image
import math
from time import sleep


def read_mat_file(file_name):
    workspace = sio.loadmat(file_name)
    data = workspace['data']
    labels = workspace['labels']
    return data, labels


def write_to_tf(files, save_path, save_filename):
    random.shuffle(files)
    patch_obj = Patches(
        img_patch_h=600, img_patch_w=600,
        stride_h=450, stride_w=450,
        label_patch_h=600, label_patch_w=600)
    print('Writing', os.path.join(save_path, save_filename + '.tfrecords'))
    tf_writer = tf.python_io.TFRecordWriter(os.path.join(save_path, save_filename + '.tfrecords'))
    num_examples = 0
    printProgressBar(0, len(files), prefix='Progress:', suffix='Complete', length=50)

    for file_n in range(0, len(files)):
        curr_train_file = str(files[file_n])
        print('Processing ' + curr_train_file)
        printProgressBar(file_n + 1, len(files), prefix='Progress:', suffix='Complete', length=50)
        data, labels = read_mat_file(file_name=curr_train_file)
        data = patch_obj.extract_patches(data)
        labels = patch_obj.extract_patches(labels)
        for i in range(data.shape[0]):
            #print(i)
            tf_serialized_example = encode(in_feat=data[i], labels=labels[i])
            tf_writer.write(tf_serialized_example)
            num_examples += 1

    out_dict = {'num_examples': num_examples}
    sio.savemat(os.path.join(save_path, save_filename + '.mat'), out_dict)
    tf_writer.close()


def run(opts_in):
    save_path = opts_in['save_path']
    main_file_path = opts_in['main_file_path']
    train_filename = opts_in['train_filename']
    valid_filename = opts_in['valid_filename']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_files = list(main_file_path.joinpath('mat').glob('Train*.mat'))
    valid_files = list(main_file_path.joinpath('mat').glob('Valid*.mat'))

    write_to_tf(files=train_files, save_path=save_path, save_filename=train_filename)
    write_to_tf(files=valid_files, save_path=save_path, save_filename=valid_filename)


#Patches
class Patches:
    def __init__(self, img_patch_h, img_patch_w, stride_h=1, stride_w=1, label_patch_h=None, label_patch_w=None):
        assert img_patch_h > 0, 'Height of Image Patch should be greater than 0'
        assert img_patch_w > 0, 'Width of Image Patch should be greater than 0'
        assert label_patch_h > 0, 'Height of Label Patch should be greater than 0'
        assert label_patch_w > 0, 'Width of Label Patch should be greater than 0'
        assert img_patch_h >= label_patch_h, 'Height of Image Patch should be greater or equal to Label Patch'
        assert img_patch_w >= label_patch_w, 'Width of Image Patch should be greater or equal to Label Patch'
        assert stride_h > 0, 'Stride should be greater than 0'
        assert stride_w > 0, 'Stride should be greater than 0'
        assert stride_h <= label_patch_h, 'Row Stride should be less than or equal to Label Patch Height'
        assert stride_w <= label_patch_w, 'Column Stride should be less than or equal to Label Patch Width'
        self.img_patch_h = img_patch_h
        self.img_patch_w = img_patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.label_patch_h = label_patch_h
        self.label_patch_w = label_patch_w
        self.img_h = None
        self.img_w = None
        self.img_d = None
        self.num_patches_img = None
        self.num_patches_img_h = None
        self.num_patches_img_w = None
        self.label_diff_pad_h = 0
        self.label_diff_pad_w = 0
        self.pad_h = 0
        self.pad_w = 0

    @staticmethod
    def read_image(input_str):
        image = np.array(Image.open(input_str))
        return image

    def update_variables(self, image):
        self.img_h = np.size(image, 0)
        self.img_w = np.size(image, 1)
        self.img_d = np.size(image, 2)

    def extract_patches_img_label(self, input_img_value, input_label_value):
        if type(input_img_value) == str:
            image = self.read_image(input_img_value)
        elif type(input_img_value) == np.ndarray:
            image = input_img_value
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)

        if type(input_label_value) == str:
            label = self.read_image(input_label_value)
        elif type(input_label_value) == np.ndarray:
            label = input_label_value
        else:
            raise Exception('Please input correct label path or numpy array')
        assert image.shape == label.shape, 'Image and Label shape should be the same'

        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
        stride_h = self.stride_h
        stride_w = self.stride_w

        self.label_diff_pad_h = math.ceil((img_patch_h - label_patch_h) / 2.0)
        self.label_diff_pad_w = math.ceil((img_patch_w - label_patch_w) / 2.0)

        image = np.lib.pad(image, ((self.label_diff_pad_h, self.label_diff_pad_h), (self.label_diff_pad_w, self.label_diff_pad_w), (0, 0)), 'symmetric')
        label = np.lib.pad(label, ((self.label_diff_pad_h, self.label_diff_pad_h), (self.label_diff_pad_w, self.label_diff_pad_w), (0, 0)), 'symmetric')

        self.update_variables(image)

        img_h = self.img_h
        img_w = self.img_w

        self.num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h + 1)
        self.num_patches_img_w = math.ceil(((img_w - img_patch_w) / stride_w + 1))
        num_patches_img = self.num_patches_img_h*self.num_patches_img_w
        self.num_patches_img = num_patches_img
        iter_tot = 0
        img_patches = np.zeros((num_patches_img, img_patch_h, img_patch_w, image.shape[2]), dtype=image.dtype)
        label_patches = np.zeros((num_patches_img, label_patch_h, label_patch_w, label.shape[2]), dtype=image.dtype)
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w, :]
                label_patches[iter_tot, :, :, :] = label[
                                                   start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                                                   start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w, :]
                iter_tot += 1

        return img_patches, label_patches

    def extract_patches(self, input_img_value):
        if type(input_img_value) == str:
            image = self.read_image(input_img_value)
        elif type(input_img_value) == np.ndarray:
            image = input_img_value
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)

        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
        stride_h = self.stride_h
        stride_w = self.stride_w

        if image.shape[0] < img_patch_h:
            self.pad_h = img_patch_h - image.shape[0]

        if image.shape[1] < img_patch_w:
            self.pad_w = img_patch_w - image.shape[1]

        image = np.lib.pad(image, (
        (self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0, 0)),
                           'symmetric')

        self.label_diff_pad_h = math.ceil((img_patch_h - label_patch_h) / 2.0)
        self.label_diff_pad_w = math.ceil((img_patch_w - label_patch_w) / 2.0)

        image = np.lib.pad(image, (
        (self.label_diff_pad_h, self.label_diff_pad_h), (self.label_diff_pad_w, self.label_diff_pad_w), (0, 0)),
                           'symmetric')

        self.update_variables(image)

        img_h = self.img_h
        img_w = self.img_w

        self.num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h + 1)
        self.num_patches_img_w = math.ceil(((img_w - img_patch_w) / stride_w + 1))
        num_patches_img = self.num_patches_img_h*self.num_patches_img_w
        self.num_patches_img = num_patches_img
        iter_tot = 0
        img_patches = np.zeros((num_patches_img, img_patch_h, img_patch_w, image.shape[2]), dtype=image.dtype)
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w, :]
                iter_tot += 1

        return img_patches

    def merge_patches(self, patches):
        img_h = self.img_h
        img_w = self.img_w
        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
        stride_h = self.stride_h
        stride_w = self.stride_w
        num_patches_img = self.num_patches_img
        assert num_patches_img == patches.shape[0], 'Number of Patches do not match'
        assert img_patch_h == patches.shape[1] or label_patch_h == patches.shape[1], 'Height of Patch does not match'
        assert img_patch_w == patches.shape[2] or label_patch_w == patches.shape[2], 'Width of Patch does not match'
        # label = 0
        # if label_patch_h == patches.shape[1] and label_patch_w == patches.shape[2]:
        #     label = 1
        image = np.zeros((img_h, img_w, patches.shape[3]), dtype=np.float)
        sum_c = np.zeros((img_h, img_w, patches.shape[3]), dtype=np.float)
        iter_tot = 0
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                if self.label_diff_pad_h == 0 and self.label_diff_pad_w == 0:
                    image[start_h:end_h, start_w:end_w, :] += patches[iter_tot, :, :, :]
                    sum_c[start_h:end_h, start_w:end_w, :] += 1.0
                else:
                    image[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w, :] += \
                        patches[iter_tot, :, :, :]
                    sum_c[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w, :] += 1.0
                iter_tot += 1

        if self.pad_h != 0 and self.pad_w != 0:
            sum_c = sum_c[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]
            image = image[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]

        if self.pad_h == 0 and self.pad_w != 0:
            sum_c = sum_c[:, self.pad_w:-self.pad_w, :]
            image = image[:, self.pad_w:-self.pad_w, :]

        if self.pad_h != 0 and self.pad_w == 0:
            sum_c = sum_c[self.pad_h:-self.pad_h, :, :]
            image = image[self.pad_h:-self.pad_h, :, :]

        # sum_c = sum_c[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]
        # image = image[self.pad_h:-self.pad_h, self.pad_w:-self.pad_w, :]
        assert (np.min(sum_c) >= 1.0)
        image = np.divide(image, sum_c)

        return image

#data_utils
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode(in_feat, labels):
    if len(labels.shape) == 2:
        labels = np.expand_dims(labels, axis=2)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'in_feat/shape': int64_list_feature(in_feat.shape),
                'in_feat/data': bytes_feature(in_feat.tostring()),
                'labels/shape': int64_list_feature(labels.shape),
                'labels/data': bytes_feature(labels.tostring())}))

    return tf_example.SerializeToString()

#Tools
def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill=chr(0x00A3)):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()



if __name__ == '__main__':
    opts = {
        #'save_path': pathlib.Path(r'P:\DCIS_Duke_Faranak\HDD\Final_IHC_model\Data\1103202020-NewdataforTF\combine_mat_P&N\curated\cv2'),
        #'main_file_path': pathlib.Path(r'P:\DCIS_Duke_Faranak\HDD\Final_IHC_model\Data\1103202020-NewdataforTF\combine_mat_P&N\curated\cv2\AugData-20200315'),
        #'train_filename': 'TrainData-IHC-CV2',
        #'valid_filename': 'ValidData-IHC-CV2'

        }

    run(opts_in=opts)