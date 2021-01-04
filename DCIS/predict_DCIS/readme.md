## DCIS segmentation using IM-Net

This pipeline will segment DCIS from the tiled image.

### Docker image 

https://hub.docker.com/repository/docker/nrypri001docker/tfcpu1p9

## Parameters for DCIS segmentation

#exp_dir-> checkpoint_path                        
#data_dir-> cws_path                               
#result_dir-> dcis_segmentation_path                                    
#result_subdir-> detection_path                                     
#tissue_segment_dir-> tissue_segmentation_result_path
#file_name_pattern -> *.svs(WSI slide extension obtained from the prefix or extension suffix)

### Command line arguments for DCIS segmentation using IM-Net

```generate_output_main_DCIS.py --exp_dir=exp_dir --data_dir=data_dir --result_dir=result_dir --tissue_segment_dir=tissue_segment_dir --file_name_pattern=file_name_pattern```

#### Note
Source code is available in this repository and tested with the packaged docker image for DCIS segmentation.
