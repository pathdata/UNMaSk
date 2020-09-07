import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#from  configs import cell_type_map_dict, setFigureObjectProperties

from scipy.stats import ttest_ind

#setFigureObjectProperties()

def computeCombinedMorisita(input_dir, output_dir=None, save_fig=False):
    
    os.makedirs(output_dir, exist_ok=True)
    voronoi_cell_count_df = pd.DataFrame()
    morisita_df=pd.DataFrame(columns=['Cell name', 'Morisita index'])
    for i,folder in enumerate(os.listdir(input_dir)):
        
             
        #
        
              
        df = pd.read_csv(os.path.join(input_dir, folder, 'voronoi_cell_count.csv'))
        if i==0:
            voronoi_cell_count_df= df
        else:
            voronoi_cell_count_df = pd.concat([voronoi_cell_count_df, df], ignore_index=True, sort=False)

    #'plot relationship here'
    voronoi_cell_count_df = voronoi_cell_count_df.drop(['voronoi_n'], axis=1)
    #voronoi_cell_count_df = voronoi_cell_count_df.rename(columns=cell_type_map_dict)
    ref_cell_name = 't'
    for cell_name in list(voronoi_cell_count_df.columns):
        
        plt.close()
        if cell_name == ref_cell_name:
            	continue
        else:
            x = voronoi_cell_count_df[ref_cell_name]
            y = voronoi_cell_count_df[cell_name]
            x, y = removeZeroCounts(x, y)
            x, y = removeZeroCounts(x, y)
            x, y = normalize_vector(x, y)
            M = getRegionMorisita(x,y)
            morisita_df.loc[len(morisita_df)] = [cell_name, np.round(M, decimals=3)]
    morisita_df.to_csv(os.path.join(output_dir, '{}_morisita_index.csv'.format(ref_cell_name)), index=False)

    morisita_df = morisita_df.set_index('Cell name')
    ax = plt.subplot(111)
    morisita_df.plot(kind='bar', ax=ax, legend=False)
    plt.xticks(rotation=70)
    plt.ylabel('Morisita index')
    plt.xlabel('Cell name')
    # plt.title('Morisita index')
    plt.tight_layout()
    # plt.box(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height() * 1.01), color='r')

    if save_fig is True:
        plt.savefig(os.path.join(output_dir, '20190524_morisita.png'), dpi=600)
    else:
        plt.show()

def computeSlideLevelMorisita(input_dir,
                              output_dir=None,
							  ref_cell_name = 'l',
                              save_fig=False):


    os.makedirs(output_dir, exist_ok=True)

    # voronoi_cell_count_df = pd.DataFrame()
    cell_names  = ['f', 't']
    morisita_df=pd.DataFrame(columns=['Slide name']+ cell_names)
    for ii,folder in enumerate(os.listdir(input_dir)):
        
              
        os.makedirs(os.path.join(output_dir,folder), exist_ok=True)
        
        voronoi_cell_count_df = pd.read_csv(os.path.join(input_dir, folder, 'voronoi_cell_count.csv'))
        
        voronoi_cell_count_df = voronoi_cell_count_df.drop(['voronoi_n'], axis=1)
        
        row  = [folder]
        for cell_name in cell_names:           
            
            if cell_name == ref_cell_name:
                continue;
            else:
                                  
                x = voronoi_cell_count_df[ref_cell_name]
                y = voronoi_cell_count_df[cell_name]
                x, y = removeZeroCounts(x, y)
                
                x, y = removeZeroCounts(x, y)
                x, y = normalize_vector(x, y)
                M = getRegionMorisita(x,y)
                print('M={}'.format(M))
                row.append(np.round(M, decimals=3))
        morisita_df.loc[len(morisita_df)] = row
                
        morisita_df.to_csv(os.path.join(output_dir, folder, '{}_slide_level_morisita_index.csv'.format(ref_cell_name)), index=False)

    # morisita_df = morisita_df.set_index('Cell name')
    plt.close()
    ax = plt.subplot(111)
    mean_series  =morisita_df.mean().round(decimals=3)
    std_dev = morisita_df.std().round(decimals=3)
    mean_series.plot(kind='bar', ax=ax, yerr = std_dev,capsize=4, legend=False)
    plt.xticks(rotation=50)
    plt.ylabel('Morisita index')
    plt.xlabel('Cell name')
    plt.title('Colocalization of tumour cells/Lymphocytes in DCIS regions')

    # plt.box(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height() * 1.01), color='r')
    plt.tight_layout()

    if save_fig is True:
        plt.savefig(os.path.join(output_dir, folder,'individual_morisita.png'), dpi=600)
    else:
        plt.show()


def removeZeroCounts(input_1, input_2, threshold=1):

    input_1  = np.array(input_1)
    input_2 = np.array(input_2)

    sum_values  = input_1 + input_2

    input_1 = input_1[sum_values >= threshold]
    input_2 = input_2[sum_values >= threshold]

    return input_1, input_2

def normalize_vector(input_1, input_2):

    input_1  = np.array(input_1)
    input_2 = np.array(input_2)

    sum_values  = input_1 + input_2

    in1_norm = input_1/sum_values
    in2_norm = input_2 / sum_values

    return in1_norm, in2_norm

def getRegionMorisita(in_1, in_2):
    num = 2 * np.sum(in_1 * in_2)
    denom = np.sum(np.power(in_2, 2)) + np.sum(np.power(in_1, 2))
    morisita_index = num / denom
    return morisita_index


def run_1():

    output_dir_home = r'D:\DCIS_2019\Morisita_DL_DuctSegm_results\2019_PureDCIS_Morisita'
    input_dir_home = r'D:\DCIS_2019\Morisita_DL_DuctSegm_results\2019_PureDCIS_Voronoi'
    computeSlideLevelMorisita(input_dir_home, output_dir=output_dir_home,  save_fig=True)


if __name__=='__main__':
    run_1()
