import argparse

#mode 'test' --model=model_HE_Inception_unet --test=Test --result=output
#mode 'train' --bs=4 --epochs=100
#mode 'WSI_test' --model=model_HE_Inception_unet --test=Test --result=output

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', dest='model', help='path to the model', default='specify the path')
    parser.add_argument('-test', '--test', dest='test', help='path to test images')
    parser.add_argument('-result', '--result', dest='result', help='path to predicted result images')
    parser.add_argument('-mode', '--mode', dest='mode',type=str, help='train or predict To perform prediction on test images prediction mode is set to test', default='test')
    parser.add_argument('-bs', '--batch_size', type=int, dest='batch_size', help='batch size of training images', default=None)
    parser.add_argument('-epochs', '--epochs', type=int, dest='epochs', help='total number of epochs',default=None)
    args = parser.parse_args()
    return args
