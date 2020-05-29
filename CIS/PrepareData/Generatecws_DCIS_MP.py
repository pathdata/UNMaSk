# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:16:05 2017

@author: pnarayanan
"""

import openslide
import os
import numpy as np
from PIL import Image
import multiprocessing as mp
import pickle

class Generatecws(object):
    def __init__(self,
                 input_dir,
                 output_dir,
                 ext,
                 num_processes,
                 exp_dir,
                 objective_power,
                 slide_dimension,
                 rescale,
                 cws_objective_value,
                 filename,
                 cws_read_size
                 ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ext = ext
        self.num_processes  = num_processes
        self.objective_power = objective_power
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)
            
    def generate_tiles(self, slide_names_name_list,process_num):
        #slide_names_name_list= [ fname for fname in os.listdir(self.input_dir) if fname.endswith(self.ext) is True]


        for s_n,slide_name in enumerate(slide_names_name_list):
            print('Process number:{}....Creating tile from slide:{}... {}/{}'.format(process_num,slide_name, s_n, len(slide_names_name_list)))
            
        
            osr=openslide.OpenSlide(os.path.join(self.input_dir,slide_name))
            # dir1=immg
            
            if os.path.exists(os.path.join(self.output_dir,slide_name)) is False:
                os.makedirs(os.path.join(self.output_dir,slide_name))
        
                  
            level=0

            ds=int(osr.level_downsamples[level])
            
            w,h=osr.level_dimensions[0]
            width=2000
            height=2000
            
            
            k=0
            for j in range(0,h,2000):   
                for i in range(0,w,2000):    
            
                    if(i+2000>w):
                        width=w-i
                    else:
                        width=2000           
                    if(j+2000>h):
                        height=h-j 
                    else:
                        height=2000  
                    ww=int(width/ds)
                    hh=int(height/ds)
                    out=osr.read_region((i,j),level,(ww,hh))
                    temp = np.array(out)
                    temp = temp[:, :, 0:3]
                    out = Image.fromarray(temp)
            
                  
                    
                    out.save(os.path.join(self.output_dir, slide_name, 'Da'+str(k)+".jpg"))
                    k += 1
    def apply_multiprocessing(self):

        l = [ fname for fname in os.listdir(self.input_dir) if fname.endswith(self.ext) is True]
        n = len(l)
        num_elem_per_process = int(np.ceil(n / self.num_processes))

        file_names_list_list = []

        for i in range(self.num_processes):
            start_ = i * num_elem_per_process
            x = l[start_: start_ + num_elem_per_process]
            file_names_list_list.append(x)

        print('{} processes created.'.format(self.num_processes))
        # create list of processes
        processes = [
            mp.Process(target=self.generate_tiles, args=(file_names_list_list[process_num], process_num)) for
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
            file_names_list = [fname for fname in os.listdir(self.input_dir) if fname.endswith(self.ext) is True]
            self.generate_tiles(file_names_list, 1)
        else:
            self.apply_multiprocessing()
            
    def get_params(self):
        
    	for slide_name in os.listdir(self.input_dir):
    		osr = openslide.OpenSlide(os.path.join(self.input_dir, slide_name))
    		w, h = osr.level_dimensions[0]
    		params['slide_dimension'] = osr.level_dimensions[0]
    		params['exp_dir']= os.path.join(self.output_dir, slide_name)
    		params['filename'] = slide_name
    		with open(os.path.join(self.output_dir, slide_name, 'param.p'), 'wb') as file:
    			pickle.dump(params, file)
                
    def get_slide_thumbnail(self):
        
        for slide_name in os.listdir(self.input_dir):
            osr = openslide.OpenSlide(os.path.join(self.input_dir, slide_name))
        
        
            openslide_obj = osr
            cws_objective_value = self.objective_power
            output_dir = self.output_dir
        
            if self.objective_power == 0:
                self.objective_power = np.int(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            slide_dimension = openslide_obj.level_dimensions[0]
            rescale = np.int(self.objective_power / cws_objective_value)
            slide_dimension_20x = np.array(slide_dimension) / rescale
            scale_h = int(slide_dimension[0]) / 1024
            thumbnail__height = slide_dimension[1] / scale_h
            thumb = openslide_obj.get_thumbnail([1024, thumbnail__height])
            thumb.save(os.path.join(output_dir, slide_name,'SlideThumb.jpg'), format='JPEG')
            thumb = openslide_obj.get_thumbnail(slide_dimension_20x / 16)
            thumb.save(os.path.join(output_dir,slide_name, 'Ss1.jpg'), format='JPEG')




if __name__ == '__main__':
    params  = {'input_dir':r'Y:\vk_Backup\DCIS_Duke\data\raw\5thbatch', ########### input dir of whole slide images
               'output_dir':r'Y:\vk_Backup\DCIS_Duke\data\cws\5thbatch',########### cws dir of tiles
               'ext':'.svs',
               'num_processes':6,
               'exp_dir': '',
			   'objective_power': 20,
               'slide_dimension': [],
			   'rescale': 1,
			   'cws_objective_value': 20,
               'filename':'',
			   'cws_read_size': (2000, 2000)
                }
    obj = Generatecws(**params)
    obj.run()
    obj.get_params()
    obj.get_slide_thumbnail()
#%%