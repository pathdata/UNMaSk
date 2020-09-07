from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
import numpy as np
import matplotlib.pyplot as plt
import cv2

import re,json,collections
import glob
import os


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def centroids1(file_name):
    image = cv2.imread(file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_br=cnts[1]
    x1=[]
    y1=[]
    for c in cnts_br:
        area = cv2.contourArea(c)
        if(area>50):
                M = cv2.moments(c)
                if(M["m00"]!=0):
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        x1.append(cX)
                        y1.append(cY)
    return x1,y1

def Contours(file_name):
    image = cv2.imread(file_name)
    cv2.imwrite("e.jpg",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_br=cnts[1]
    area1=[]
    for c in cnts_br:
        area = cv2.contourArea(c)
        area1.append(area)
    m_area=max(area1)
    g=area1.index(m_area)

    return cnts_br[g]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def write_voronoi_detail(dir_output, file_name, vert):
    if os.path.exists(os.path.join(dir_output, file_name)) is False:
        os.makedirs(os.path.join(dir_output, file_name))

    vor = get_voronoi_polygon_vertices(vert)
    with open(os.path.join(dir_output, file_name,
                           'voronoi_data.json'), 'w') as fp:
        json.dump(vor, fp)


def get_voronoi_polygon_vertices(vertices):
    voronoi_data = dict()
    #    reg_iter = 0
    for i, vertices_i in enumerate(vertices):
        voronoi_data['voronoi_{}'.format(i)] = vertices_i
    #        reg_iter +=1

    return voronoi_data


class Spatial_data_extract_S1(object):

    def __init__(self,
                 input_slide_dir,
                 obj_dir,
                 tissue_mask_dir,
                 output_dir,
                 ext):
        self.input_slide_dir = input_slide_dir
        self.obj_dir = obj_dir
        self.tissue_mask_dir = tissue_mask_dir
        self.output_dir = output_dir
        self.ext = ext
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)



    def data_extract(self):

        list_img = os.listdir(self.input_slide_dir)
        list_obj_mask = os.listdir(self.obj_dir)
        list_obj_Tissue = os.listdir(self.tissue_mask_dir)
        list_img = sorted(list_img, key=natural_key)
        list_obj_mask = sorted(list_obj_mask, key=natural_key)
        list_obj_Tissue = sorted(list_obj_Tissue, key=natural_key)

        voronoi_dir = 'voronoi'

        if os.path.exists(os.path.join(self.output_dir, voronoi_dir)) is False:
            os.makedirs(os.path.join(self.output_dir, voronoi_dir))

        for p1 in range(len(list_img)):
            file_name = list_img[p1]
            img = cv2.imread(os.path.join(self.input_slide_dir, list_img[p1]))
            img_orig = img.copy()

            file1 = os.path.join(self.obj_dir, list_obj_mask[p1])
            file2 = os.path.join(self.tissue_mask_dir, list_obj_Tissue[p1])

            x1 = []
            y1 = []
            points11 = []
            x1, y1 = centroids1(file1)

            for i in range(0, len(x1)):
                points11.append([x1[i], y1[i]])

            points = np.array(points11)

            vor = Voronoi(points)

            regions, vertices = voronoi_finite_polygons_2d(vor)
            cnts_img_t = Contours(file2)
            point_t = []

            for c1 in cnts_img_t:

                point_t.append([c1[0][0], c1[0][1]])
            point_t = np.array(point_t)

            pts = MultiPoint([Point(i) for i in point_t])
            mask = pts.convex_hull
            print("mask=", mask.bounds)

            new_vertices = []
            a = 0
            for region in regions:
                print("a=", a)
                a = a + 1
                polygon = vertices[region]
                shape = list(polygon.shape)
                shape[0] += 1
                p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
                print("p=", p.bounds)
                print("lk=", int(p.length))
                l1 = int(p.length)
                if (l1 > 0):
                    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
                    new_vertices.append(poly)
                    new_vertices.append(poly)

            for p1 in new_vertices:

                pts = np.array(p1, np.int32)
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(img, [pts], True, (0, 0, 0), 3)
            for p1 in points11:

                cv2.circle(img, (p1[0], p1[1]), 13, (0, 255, 0), cv2.FILLED, cv2.LINE_AA, 0)
                #
            cv2.imwrite(os.path.join(self.output_dir,voronoi_dir,file_name), img)

            new_vert = []
            for i in range(len(new_vertices)):
                new_vert.append(new_vertices[i].tolist())

            write_voronoi_detail(self.output_dir, file_name, new_vert)




if __name__ == '__main__':


    params = {'input_slide_dir': r'D:\UNET_Experiments\voronoi_DUKE\Images',      # input slide SS1_dir
              'obj_dir': r'D:\UNET_Experiments\voronoi_DUKE\obj_mask',            # DCIS output mask from matlab
              'tissue_mask_dir': r'D:\UNET_Experiments\voronoi_DUKE\Tissue_mask', # tissuemask
              'output_dir': r'D:\UNET_Experiments\voronoi_DUKE',                  # output dir
              'ext': '.jpg',
              }

    obj = Spatial_data_extract_S1(**params)
    obj.data_extract()


