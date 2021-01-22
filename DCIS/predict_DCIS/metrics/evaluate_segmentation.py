import numpy as np
import os
import cv2
import pandas as pd
#import matplotlib.pyplot as plt

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise ("DiffDim: Different dimensions of matrices!")


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width



def run_pixel_accuracy(pd, gt):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''



    #check_size(eval_segm, gt_segm)

    sum_n_ii = 0
    sum_t_i = 0

    sum_n_ii += np.sum(np.logical_and(pd, gt))
    sum_t_i += np.sum(gt)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0

    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def get_dice_traditional(gt, pd):
    """
        Traditional Dice co-efficient
    """
    # generate to binary 1st
    gt = np.copy(gt)
    pred = np.copy(pd)
    gt[gt > 0] = 1
    pred[pred > 0] = 1
    intersection = gt * pred
    denom = gt + pred
    return 2.0 * np.sum(intersection) / np.sum(denom)

def get_dice_modified(gt, pd):
    gt = np.copy(gt)
    pred = np.copy(pd)
    gt_id = list(np.unique(gt))
    pred_id = list(np.unique(pred))
    # remove background index 0
    gt_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in gt_id:
        t_mask = np.array(gt == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pd == p, np.uint8)
            intersect = p_mask * t_mask          
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += (t_mask.sum() + p_mask.sum())
    return 2 * total_intersect / total_markup
#####

def Panoptic_quality(gt, pd):
        TP = 0
        FP = 0
        FN = 0
        sum_IOU = 0
        matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
        # It will also save IOU of the matched instance in [indx][1]

        # Find matched instances and save it in a dictionary
        for i in np.unique(gt):
            #print(i)
            if i == 0:
                pass
            else:
                temp_image = np.array(gt)
                temp_image = temp_image == i
                matched_image = temp_image * pd

                for j in np.unique(matched_image):
                    #print(j)
                    if j == 0:
                        pass
                    else:
                        pred_temp = pd == j
                        intersection = sum(sum(temp_image * pred_temp))
                        union = sum(sum(temp_image + pred_temp))
                        IOU = intersection / union
                        if IOU > 0.5:
                            matched_instances[i] = j, IOU
                            # print(matched_instances)

        # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

        pred_indx_list = np.unique(pd)
        pred_indx_list = np.array(pred_indx_list[1:])

        # Loop on ground truth instances
        for indx in np.unique(gt):
            if indx == 0:
                pass
            else:
                if indx in matched_instances.keys():
                    pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == [indx][0]))
                    TP = TP + 1
                    sum_IOU = sum_IOU + matched_instances[indx][1]
                else:
                    FN = FN + 1
        FP = len(np.unique(pred_indx_list))
        PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)


        return PQ




if __name__ == '__main__':


    params = {

    'gt_file_path': r'E:\EVAL\gt',
    'pred_file_path':r'E:\EVAL\pd'
    }

    pred_file_path = params['pred_file_path']

    gt_file_path = params['gt_file_path']

    eval_score = pd.DataFrame(columns=['slidename', 'pixel_accuracy', 'dice'])

    for pred_img in os.listdir(pred_file_path):

        eval_segm = cv2.imread(os.path.join(pred_file_path, os.path.splitext(pred_img)[0]+'.png'), 0)

        gt_segm = cv2.imread(os.path.join(gt_file_path, os.path.splitext(pred_img)[0]+'.jpg'), 0)

        if np.amax(eval_segm) > 1:
            eval_segm[eval_segm < 127] = 0
            eval_segm[eval_segm >= 127] = 1
            eval_segm = eval_segm.astype(np.uint8)

        if np.amax(gt_segm) > 1:
            gt_segm[gt_segm < 127] = 0
            gt_segm[gt_segm >= 127] = 1
            gt_segm = gt_segm.astype(np.uint8)

        pd = eval_segm
        gt = gt_segm
        pix_acc = run_pixel_accuracy(pd, gt)
        dice1 = get_dice_traditional(gt, pd)
        dice2 = get_dice_modified(gt, pd)
        PQ = Panoptic_quality(gt, pd)
        
        eval_score.columns = ['slidename', 'pixel_accuracy', 'dice']
        eval_score = eval_score.append({'slidename': pred_img, 'pixel_accuracy': pix_acc, 'dice': dice2}, ignore_index=True)
    eval_score.to_csv('E:\\EVAL\\Eval_Score_test.csv',
                 encoding='utf-8', index=False)
