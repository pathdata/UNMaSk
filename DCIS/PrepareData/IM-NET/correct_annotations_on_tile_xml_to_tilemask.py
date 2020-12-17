from xml.dom import minidom
import cv2
import numpy as np
import re
import os


def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


class Generate_tileannotation_on_cws(object):

    def __init__(self,
                 input_dir,
                 mask_dir,
                 ext):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.ext = ext
        if os.path.exists(self.mask_dir) is False:
            os.makedirs(self.mask_dir)

    def generate_mask_of_annotated_tiles(self):

        file_names_list = [fname for fname in os.listdir(self.input_dir) if fname.endswith(self.ext) is True]


        for files in file_names_list:

            if files.endswith(".jpg"):

                name = os.path.splitext(files)[0]
                img_name = files
                xml_name = name + ".xml"

                mydoc = minidom.parse(os.path.join(self.input_dir, xml_name))

                img = cv2.imread(os.path.join(self.input_dir, img_name))
                blank_image = np.zeros((img.shape), np.uint8)

                Region = mydoc.getElementsByTagName('Region')

                for Reg in Region:
                    X = []
                    Y = []

                    Vertex = Reg.getElementsByTagName("Vertex")
                    for Vert in Vertex:
                        X.append(int(round(float(Vert.getAttribute("X")))))
                        Y.append(int(round(float(Vert.getAttribute("Y")))))

                    points = []
                    for i3 in range(len(X)):
                        points.append([int(X[i3]), int(Y[i3])])
                    pts = np.array(points, np.int32)
                    cv2.drawContours(blank_image, [pts], 0, (255, 255, 255), -1)
                cv2.imwrite(os.path.join(self.mask_dir, img_name), blank_image)

if __name__ == '__main__':
    params  = {'input_dir': r'D:\2019_Rose\AZ _Pres_Jesu\2020_XP\annotation_xml',       # input slide directory
               'mask_dir': r'D:\2019_Rose\AZ _Pres_Jesu\2020_XP\annotation_mask',       # output mask directory
               'ext': '.jpg',
               }
    obj = Generate_tileannotation_on_cws(**params)
    obj.generate_mask_of_annotated_tiles()






