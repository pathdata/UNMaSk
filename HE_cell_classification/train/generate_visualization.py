import pathlib

from sccnn_classifier import featuremap_visualization

if __name__ == '__main__':
    opts = {
        'exp_dir': str(pathlib.Path(
            r'D:\2019_TA_SCCNN\20191105_SCCNNClassifier\SCCNN')),
        'data_dir': str(pathlib.Path(
            r'D:\2019_TA_SCCNN\annotations\celllabels\TA030_007-2015-12-07_16.37.59.ndpi/')),
        'cws_dir': str(pathlib.Path(
            r'D:\2019_TA_SCCNN\cws_TA\TA030_007-2015-12-07_16.37.59.ndpi/'))
    }

    featuremap_visualization.run(opts_in=opts)
