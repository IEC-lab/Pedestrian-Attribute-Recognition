# config.py
import os.path

VOCroot = '/home/lyf/data/VOCdevkit'

VOC_Config = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [21, 45, 99, 153, 207, 261],

    'max_sizes' : [45, 99, 153, 207, 261, 315],

    'aspect_ratios' : [[0.4, 1.0, 1.5, 2.5],
                       [0.4, 1.0, 1.6, 2.6],
                       [0.5, 1.1, 1.6, 2.6],
                       [0.5, 1.1, 1.6, 2.6],
                       [0.5, 1.1, 1.6, 2],
                       [0.7, 1.4]],

    'max_ratios' : [0.8, 0.8, 0.8, 0.9, 1, 1],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

