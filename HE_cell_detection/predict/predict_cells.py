from inference_architecture.celldetect import get_Inception_unet
from skimage import io
import numpy as np
import os
import keras.backend as K
from  scipy.ndimage.morphology import binary_fill_holes
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import cv2
import multiprocessing as mp
from skimage.morphology import dilation,,disk
import os
from scipy.ndimage import measurements
import pandas as pd
from configs import config
import re

def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class DetectCells(object):
    def __init__(self, model_weight_dir,
                 input_dir,
                 tissue_mask_dir,
                 output_dir,
                 detection_annotation_dir,
                 num_processes=1,
                 cws_mask = None,
                 q1=11,
                 q3=40,
                 count_loss_used=False,
                 patch_size=224,
                 normalization='regular',
                 stride=None,
                 overlap=False):
        self.model_weight_dir=model_weight_dir
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.tissue_mask_dir=tissue_mask_dir
        self.cws_mask  = cws_mask
        self.detection_annotation_dir  = detection_annotation_dir

        self.num_processes = num_processes
        self.patch_size=patch_size
        self.overlap=overlap

        self.normalization =normalization
        self.count_loss_used = count_loss_used
        # self.area_threshold  = area_threshold
        self.q1 = q1
        self.q3 = q3

        self.area_threshold = self.compute_area_threshold_from_quartiles()

        if overlap == False or stride == None:
            self.stride = patch_size
        else:
            self.stride = stride
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)
    def compute_area_threshold_from_quartiles(self):
        iqd = self.q3 - self.q1
        return self.q3 + iqd

    def dice_coef(self, y_true, y_pred):
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)

    def count_loss(self, y_true, y_pred):
        #    y_pred = K.round(y_pred)
        smooth = 1
        mean_abs_diff = K.mean(K.abs(y_pred - y_true))
        return 1 - 1 / (smooth + mean_abs_diff)  # 1- 1/K.exp(sum_square_diff)

    def eval_tiles(self, slide_names_list, p_n):
        print('eval_tiles')
        l1 = slide_names_list  # os.listdir(self.input_dir)
        n1 = len(l1)
        model = get_Inception_unet(self.model_weight_dir)
        
        for ii, folder in enumerate(l1):
            if folder.endswith('.ini') is True:
                continue
            input_im_folder = os.path.join(self.input_dir, folder)
            output_folder = os.path.join(self.output_dir, folder)    # can be used to save binary output
            if os.path.exists(output_folder) is False:
               os.makedirs(output_folder)
            else:
                pass
            seg_output_folder = os.path.join(self.detection_annotation_dir, folder)
            if os.path.exists(seg_output_folder) is False:
                os.makedirs(seg_output_folder)
            cell_output_folder = os.path.join(self.detection_annotation_dir, 'AnnotatedCellsCoord', folder)
            cell_class_output_folder = os.path.join(self.detection_annotation_dir, 'AnnotatedCellsDetectCoord', folder)
            if os.path.exists(cell_output_folder) is False:
                os.makedirs(cell_output_folder)

            if os.path.exists(cell_class_output_folder) is False:
                os.makedirs(cell_class_output_folder)
            l2 = os.listdir(input_im_folder)

            l2 = sorted(l2, key=natural_key)
            n2 = len(l2)
            for jj, image in enumerate(l2):
                print('process number:{}, processing, Slide:{}/{}, Tile:{}/{}'.format(p_n + 1, ii + 1, n1, jj + 1,n2))

                'output file names'
                annotated_im_name = os.path.join(seg_output_folder, image)
                cell_detection_map_name = os.path.join(seg_output_folder, image)
                csv_filename = os.path.join(cell_output_folder, image.split('.')[0] + '.csv')
                detect_csv_filename = os.path.join(cell_class_output_folder, image.split('.')[0] + '.csv')

                df = pd.DataFrame(columns=['X', 'Y', 'Area'])
                dfD = pd.DataFrame(columns=['V1', 'V2', 'V3'])
                fname = os.path.join(input_im_folder, image)
                if (image.endswith('.jpg') and image.startswith('Da')) is False:
                    continue
                im = io.imread(fname)
                'cell mask'
                if self.cws_mask is not None:
                    im_mask  = io.imread(os.path.join(self.cws_mask, folder, image))

                    if np.sum(im_mask)==0:
                        if np.ndim(im_mask) == 3:
                            # io.imsave(annotated_im_name, im)
                            io.imsave(cell_detection_map_name, im_mask[:,:,1])
                        else:
                            print(0)
                            io.imsave(annotated_im_name, im)
                            io.imsave(cell_detection_map_name, im_mask)
                        io.imsave(annotated_im_name, im)
                        df['X'] = []
                        df['Y'] = []
                        df['Area'] = []
                        df.to_csv(csv_filename, index=False)

                        dfD['V1'] = []
                        dfD['V2'] = []
                        dfD['V3'] = []
                        dfD.to_csv(csv_filename, index=False)

                        continue

                pad_size = 2 * self.patch_size
                n = 2 * pad_size
                im_pad = np.zeros((n + im.shape[0], n + im.shape[1], 3), dtype='uint8')
                n = int(n / 2)
                im_pad[n:im.shape[0] + n, n:im.shape[1] + n, :] = im
                label = np.zeros(im_pad.shape[:2])
                if self.normalization == 'regular':
                    im_pad = im_pad * 1.0 / 255
                elif self.normalization == 'central':
                    im_pad = (im_pad - 128) * 1.0 / 128
                else:
                    pass
                PP = 12
                SS = int(pad_size/ 2)-PP
                # row_start_ = SS
                row_end = im_pad.shape[0] - int(self.patch_size/ 2)+PP
                col_start_ = SS
                col_end = im_pad.shape[1] - int(self.patch_size/ 2)+PP
                r = SS
                c = SS
                while r < row_end:
                    c = col_start_
                    while c < col_end:
                        r_start = r - SS
                        c_start = c - SS
                        p_image = im_pad[r_start:r_start + self.patch_size, c_start:c_start + self.patch_size, :]
                        p_image = np.expand_dims(p_image, axis=0)
                        # pred = model.predict(p_image)
                        if self.count_loss_used is True:
                            pred, cell_count = model.predict(p_image)
                        else:
                            pred = model.predict(p_image)
                        pred = np.squeeze(pred)
                        label[r_start+PP:r_start + self.patch_size-PP, c_start+PP:c_start + self.patch_size-PP] = pred[PP:pred.shape[0]-PP, PP:pred.shape[1]-PP]

                        c = c + self.stride-2*PP
                    r = r + self.stride-2*PP
                label = label[n:im.shape[0] + n, n:im.shape[1] + n]
                #io.imsave(cell_detection_map_name, label)
                label_corrected  = self.doCellSplitting(label)
                #cell_detection_map_name = os.path.join(output_folder, image)
                label = (255 * label_corrected).astype('uint8')
                io.imsave(cell_detection_map_name, label)

                X, Y, Area = self.get_cell_center(label_corrected)
                marked_im = self.mark_cell_center(im, X, Y)
                annotated_im_name = os.path.join(seg_output_folder, image)
                io.imsave(annotated_im_name, marked_im)
                df['X'] = X
                df['Y'] = Y
                df['Area'] = Area
                df.to_csv(csv_filename, index=False)

                ### Add dfD ### details


                dfD['V1'] = ['None'] * len(X)
                dfD['V2'] = X
                dfD['V3'] = Y
                dfD = dfD.loc[dfD.ne(0).all(axis=1)]
                dfD.to_csv(detect_csv_filename,
                                  index=False)




    def doCellSplitting(self, im):

        im = (im > 0.9) * 1
        im1 = binary_fill_holes(im)
        labeled_im = measurements.label(im1)[0]
        L = list(np.unique(labeled_im))
        c_mass = measurements.center_of_mass(im1, labels=labeled_im, index=L[1:])
        Area = []
        for index, center in zip(L[1:], c_mass):
            Area.append(np.sum(labeled_im == index))
        Area = np.array(Area)
        im_large_obj = np.zeros(im.shape, dtype='uint8')
        obj_l = np.array(L[1:])
        large_obj_label = obj_l[Area > self.area_threshold]
        small_obj_label = obj_l[Area <= self.area_threshold]
        small_obj_image = (np.isin(labeled_im, list(small_obj_label))) * 1
        large_obj_image = (np.isin(labeled_im, list(large_obj_label))) * 1
        im_ = np.zeros(im.shape)
        X, Y = self.splitCells(large_obj_image)
        im = self.put_markers_cv(X, Y, im_)

        return im + small_obj_image
    def splitCells(self, x):
        distance = ndi.distance_transform_edt(x)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7, 7)), labels=x)
        local_maxi = local_maxi * 1

        local_maxi_dilated = dilation(local_maxi, disk(2))
        local_maxi_label = measurements.label(local_maxi_dilated)[0]
        L = list(np.unique(local_maxi_label))
        c_mass = measurements.center_of_mass(local_maxi_dilated, labels=local_maxi_label, index=L[1:])

        X = []
        Y = []
        for center in c_mass:
            X.append(int(center[1]))
            Y.append(int(center[0]))
        return (X, Y)
    def put_markers_cv(self, X, Y, im):
        r = 2

        for i in range(len(X)):
            cv2.circle(im, (X[i], Y[i]), r, color=(1, 1, 1), thickness=-1)

        return (im > 0.8) * 1
    def get_cell_center(self, seg_image):
        im = seg_image> 0.9 * 1
        im1 = binary_fill_holes(im)
        labeled_im = measurements.label(im1)[0]
        L  = list(np.unique(labeled_im))
        c_mass = measurements.center_of_mass(im1, labels  = labeled_im, index  = L[1:])
        X = []
        Y = []
        Area  = []
        for index, center in zip(L[1:],c_mass):
            area = np.sum(labeled_im==index)
            if area > config['area_threshold']:
                X.append(int(center[1]))
                Y.append(int(center[0]))
                Area.append(area)
        return X, Y, Area

    def mark_cell_center(self, im,X, Y):
        r = 3
        X = list(X)  # row
        Y = list(Y)  # col
        for i in range(len(X)):
            cv2.circle(im, (X[i], Y[i]), r, color=(0, 255, 0), thickness=-1)


        return im

    def apply_multiprocessing(self):
        
        l = os.listdir(self.input_dir)

        
        print(l)
        n = len(l)
        print(n)
        num_elem_per_process = int(np.ceil(n / self.num_processes))

        file_names_list_list = []

        for i in range(self.num_processes):
            start_ = i * num_elem_per_process
            x = l[start_: start_ + num_elem_per_process]
            file_names_list_list.append(x)

        print('{} processes created.'.format(self.num_processes))
        
        # create list of processes
        processes = [
            mp.Process(target=self.eval_tiles, args=(file_names_list_list[process_num], process_num)) for
            process_num in range(self.num_processes)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
        print('All Processes finished!!!')
        
    def run(self):
        if self.num_processes==1:
            file_names_list = [fname for fname in os.listdir(self.input_dir)]
            print(file_names_list)
            self.eval_tiles(file_names_list, 1)
        else:
            self.apply_multiprocessing()

if __name__=='__main__':

    home = r'results'                                      # result directory
    checkpoints_filepath = r'model'                        # model directory



    params = {'model_weight_dir': os.path.join(checkpoints_filepath, 'he_model.h5'),        # model name

              'input_dir':  r'cws',                   # cws_path
              'cws_mask': None,
              'tissue_mask_dir': r'',
              'output_dir': os.path.join(home, 'CellDetection'),
              'detection_annotation_dir': os.path.join(home, 'CellsXY-Area'),
              'normalization': 'regular',
              'num_processes': 1,
              'count_loss_used': False
              }
    obj = DetectCells(**params)
    obj.run()