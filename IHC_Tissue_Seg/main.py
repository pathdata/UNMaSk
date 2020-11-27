import os
import numpy as np
from scripts.predict_IHC_Inception_unet import *
from scripts.predict_parser import *
from scripts.train_IHC_BLInception_unet import *
#import openslide

# command line usage arguments
#  main.py --model=model_IHC_Inception_unet --test=TestImages --result=output
#  main.py --bs=4 --epochs=100 --mode=train

if __name__=='__main__':

    print('Running tissue segmentation')
    args = get_parsed_arguments()
    print(args)


    if args.mode == 'test':

        print('Prediction on test Images')


        model = args.model

        test_image_path = args.test
        result_dir = args.result
        mode = args.mode
        predict_tissue(model, test_image_path, result_dir)

    elif args.mode == 'train':

        bs = args.batch_size

        epochs = args.epochs

        H = train(bs, epochs)
        plot_training_curves(output_dir='./', H=H, epochs=epochs)



