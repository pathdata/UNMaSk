import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, dest='batch_size', help='batch size of training images')
    parser.add_argument('-epochs', '--epochs', type=int, dest='epochs', help='total number of epochs')
    args = parser.parse_args()
    return args
