import os
import numpy as np
from scripts.predict_HE_Inception_unet import *
from scripts.predict_parser import *
from scripts.train_HE_BLInception_unet import *
import openslide

##### Important Usage instructions #####
# command line usage arguments

# --model=model_HE_Inception_unet --test=Test --result=output mode=test
# --bs=4 --epochs=100 mode=train
# --model=model_HE_Inception_unet --test=Test --result=output mode=WSI_test

##### Important Usage instructions #####


if __name__=='__main__':

    print('Running tissue segmentation')
    args = get_parsed_arguments()


    if args.mode=='test':

        print('Prediction on test Images')


        model = args.model

        test_image_path = args.test
        result_dir = args.result
        mode = args.mode
        predict_tissue(model, test_image_path, result_dir)

    elif args.mode=='train':

        epochs = args.epochs

        H = train(args.batch_size, args.epochs)
        plot_training_curves(output_dir='./history', H=H, epochs=epochs)

    else:
        args.mode = 'WSI_test'

        WSI_dir = r'WSI'
        ext = '.ndpi'
        objective_power = 20

        WSI_test_output_dir = r'WSI_output'

        if not os.path.exists(WSI_test_output_dir):
            os.makedirs(WSI_test_output_dir)

        file_name_list = [fname for fname in os.listdir(WSI_dir) if fname.endswith(ext) is True]

        for slide_name in file_name_list:

            osr = openslide.OpenSlide(os.path.join(WSI_dir, slide_name))

            openslide_obj = osr
            cws_objective_value = objective_power
            output_dir = WSI_test_output_dir

            if objective_power == 0:
                objective_power = np.int(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            slide_dimension = openslide_obj.level_dimensions[0]
            rescale = np.int(objective_power / cws_objective_value)
            slide_dimension_20x = np.array(slide_dimension) / rescale
            thumb = openslide_obj.get_thumbnail(slide_dimension_20x / 16)
            thumb.save(os.path.join(WSI_test_output_dir, slide_name+ '_Ss1.jpg'), format='JPEG')
            model = args.model
            test_image_path = args.test
            result_dir = args.result
            predict_tissue(model, WSI_test_output_dir, result_dir)


