import numpy as np
import pickle
import os
import glob
import cv2
import re



def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='='): #chr(0x00A3)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print()

def get_SS1_dimension_image_from_cws_resolution(cws_folder,annotated_dir,output_dir,scale):

    wsi_files = sorted(glob.glob(os.path.join(cws_folder, '*.ndpi')))

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    for wsi in range(0, len(wsi_files)):

        filename = wsi_files[wsi]

        param = pickle.load(open(os.path.join(filename, 'param.p'), 'rb'))

        slide_dimension = np.array(param['slide_dimension']) / param['rescale']

        slide_w, slide_h = slide_dimension
        cws_w, cws_h = param['cws_read_size']

        divisor_w = np.ceil(slide_w / cws_w)
        divisor_h = np.ceil(slide_h / cws_h)

        w, h = int(slide_w / scale), int(slide_h / scale)
        print('%s, Ss1 size: %i,%i'%(os.path.basename(filename),w,h))
        img_all = np.zeros((h, w, 3))

        drivepath, imagename = os.path.split(wsi_files[wsi])
        annotated_dir_i = os.path.join(annotated_dir, imagename)
        images = sorted(os.listdir(annotated_dir_i), key=natural_key)
        printProgressBar(0, len(images), prefix='Progress:', suffix='Complete', length=50)

        for i in images:
            cws_i = int(re.search(r'\d+', i).group())
            h_i = np.floor(cws_i / divisor_w) * cws_h
            w_i = (cws_i - h_i / cws_h * divisor_w) * cws_w

            h_i = int(h_i / scale)
            w_i = int(w_i / scale)
            # print(cws_i, w_i, h_i)

            img = cv2.imread(os.path.join(annotated_dir_i,i))

            img = cv2.resize(img, (int(img.shape[1]/16), int(img.shape[0]/16)))

            img_all[h_i : h_i + int(img.shape[0]), w_i : w_i + int(img.shape[1]),:] = img

            printProgressBar(cws_i, len(images), prefix='Progress:',
                             suffix='Completed for %s'%i, length=50)

            if w_i + cws_w / scale > w:
                cv2.imwrite(os.path.join(output_dir,imagename + ".png"), img_all)

if __name__ == '__main__':

    params = {
                'cws_folder' : os.path.normpath(r'D:\TF_TISSUE\HE_Tissue_seg\test_stitch_low_res\cws'),
                'annotated_dir' : r'D:\TF_TISSUE\HE_Tissue_seg\test_stitch_low_res\mask',
                'output_dir' : r'D:\TF_TISSUE\HE_Tissue_seg\test_stitch_low_res\low_res',
                'scale' : 16
               }

    get_SS1_dimension_image_from_cws_resolution(params['cws_folder'],params['annotated_dir'],params['output_dir'],params['scale'])
