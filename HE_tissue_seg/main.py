import os
import numpy as np
from scripts.predict_HE_Inception_unet import *
from scripts.predict_parser import *
from scripts.train_HE_BLInception_unet import *
#import openslide

# command line usage arguments
# --model=model_HE_Inception_unet --test=Test --result=output --mode=test
# --bs=4 --epochs=100 --mode=train

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



