

import io
import json
from skimage import io
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
import pandas as pd
import os



im_dir = r'D:/DCIS_2019/Morisita_DL_DuctSegm_results/2019_PureDCIS_Tissuemask'
voronoi_dir = r'D:/DCIS_2019/Morisita_DL_DuctSegm_results/2019_PureDCIS_Voronoi'
cells_dir = r'D:/DCIS_2019/Morisita_DL_DuctSegm_results/2019_PureDCIS_cellPos_csv'
# cmap = plt.set_cmap('Reds')

def computevoronoicellcount(cell_names):

    for ii, file_name in enumerate(os.listdir(im_dir)):
        print(file_name)

        slide_cell_count_df  = pd.DataFrame(columns=[['voronoi_n']+ cell_names])

        print('processing, file:{},..... {}/{}'.format(os.path.splitext(file_name)[0],ii + 1,len(os.listdir(im_dir))))

        cell_pos_df = pd.read_csv(os.path.join(cells_dir, os.path.splitext(file_name)[0]+'.csv'))
        cell_pos_df = cell_pos_df[cell_pos_df['class']!='notcell']
        #print(os.path.join(voronoi_dir, os.path.splitext(file_name)[0]+'.jpg'))

        voronoi_file = os.path.join(voronoi_dir, file_name,'voronoi_data.json')
        print(voronoi_file)
        if os.path.isfile(voronoi_file) is True:
            with open(voronoi_file, 'r') as file:
                voronoi_data = json.load(file)
        else:
            print('no voronoi data provided')
            continue
        for i in range(len(voronoi_data)):
            poly_vertices = voronoi_data['voronoi_{}'.format(i)]

            polygon_cell_count_df = getCellCountInsidePolygon(df=cell_pos_df,
                                                               poly_vert=poly_vertices)
            counts = ['voronoi_{}'.format(i)]
            for cell_name in cell_names:
                if cell_name in list(polygon_cell_count_df.columns):
                    counts.append(polygon_cell_count_df.loc['N',cell_name])
                else:
                    counts.append(0)
            n = len(slide_cell_count_df)
            slide_cell_count_df.loc[n]=counts

        slide_cell_count_df.to_csv(os.path.join(voronoi_dir, os.path.splitext(file_name)[0]+'.jpg', 'voronoi_cell_count.csv'), index=False)


def getCellCountInsidePolygon(df, poly_vert):
    polygon = Polygon(poly_vert)

    is_inside_polygon = df.apply(lambda row: polygon.contains(Point(row['x'], row['y'])), axis=1)

    df_inside_poly = df[is_inside_polygon]

    grouped_df = df_inside_poly.groupby(by='class').count().reset_index().set_index('class')[['x']].rename(
        columns={'x': 'N'}).transpose()
    return grouped_df


if __name__=='__main__':
    cell_names  = [ 'f', 'l', 't', 'o']
    computevoronoicellcount(cell_names=cell_names)
