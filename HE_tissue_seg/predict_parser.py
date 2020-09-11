import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', dest='model', help='path to the model', default='specify the path')
    parser.add_argument('-test', '--test', dest='test', help='path to test images')
    parser.add_argument('-result', '--result', dest='result', help='path to predicted result images')
    args = parser.parse_args()
    return args
