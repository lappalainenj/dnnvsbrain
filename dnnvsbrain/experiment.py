"""
TODO: Attribute docs.
"""
from pathlib import Path
import json
import os

import numpy as np
from matplotlib.colors import to_hex
from PIL import Image
import scipy.io as sio

root_path = Path(os.path.realpath(__file__)).parent.parent.parent

json_path = root_path / 'dnnvsbrain/json'
wnids_path = root_path / 'dnnvsbrain/wnids'

image_path = root_path / 'data/stimuli'
mat_path = root_path / 'data/mat'
misc_path = root_path / 'data/misc'

fig_path = root_path / 'figures'


class Experiment:
    """An abstraction of definitions per experiment that are partially required
    and partially optional for features of the rdm package"""

    def __init__(self):
        super(Experiment, self).__init__()
        self.labels = {}
        self.image_ids = []
        self.num_labels = None
        self.num_images_per_label = 0
        self.label_pretty = {}
        self.label_abbr = {}
        self.label_colors = {}
        self.leaf_colors = {}
        self.leaf_default_color = "#293133"
        self.correlation_colors = {}
        self.image_names = ''
        self.image_path = ''
        self.mask = {}
        self.mask_keys = []
        self._mask_keys()
        self.norm = {}
        self.fig_path = ''

    def definitions(self):
        return self.__dict__

    def load_lines(self, path, file):
        with open(path / file, 'r') as f:
            content = f.read().splitlines()
        return content

    def load_json(self, path, file):
        with open(path / file, 'r') as f:
            content = json.load(f)
        return content

    def update_leaf_colors(self):
        result = []
        for color in self.label_colors.values():
            for i in range(self.num_images_per_label):
                result.append(color)
        self.leaf_colors = {i: color for i, color in enumerate(result)}

    def _abbreviate(self, labels):
        result = {}
        for label in labels:
            result[label] = label[0:2].capitalize()
        return result

    def _pretty(self, labels):
        result = {}
        for label in labels:
            result[label] = label.replace('_', ' ').capitalize()
        return result

    def default_label_colors(self, rs=50):
        rs = np.random.RandomState(rs)
        colors = ['#6A5857',
                  '#B25B43',
                  '#D78A3F',
                  '#A2D541',
                  '#8DAD5C',
                  '#7BBC9B',
                  '#849BBB',
                  '#7257A8',
                  '#9D4CD3',
                  '#BE5786']
        for i, label in enumerate(self.labels):
            if i < len(colors):
                self.label_colors[label] = colors[i]
            else:
                self.label_colors[label] = to_hex(rs.rand(3, ))

    def default_correlation_colors(self, rs=50):
        rs = np.random.RandomState(rs)
        colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#64b5cd']
        for i, key in enumerate(self.mask_keys):
            if i < len(colors):
                self.correlation_colors[key] = colors[i]
            else:
                self.correlation_colors[key] = to_hex(rs.rand(3, ))

    def _mask_keys(self):
        for i, (key, rois) in enumerate(self.mask.items()):
            self.mask_keys.append(key)
            if isinstance(rois, dict):
                for key in rois:
                    self.mask_keys.append(key)

    def _num_images(self):
        return sum([self.num_images_per_label for label in self.labels])


class PytorchNNExperiment(Experiment):

    def __init__(self):
        super(PytorchNNExperiment, self).__init__()
        self.nnlabels = self.load_json(json_path, 'labels.json')
        self.nnlabels = {int(key): value for key, value in self.nnlabels.items()}
        self.label_from_wnid = self.load_lines(wnids_path, 'synset_wnids.txt')
        self.label_from_wnid = [wn.split('\t') for wn in self.label_from_wnid]
        self.label_from_wnid = {wn[0]: wn[1] for wn in self.label_from_wnid}
        self.wnid_from_label = self.load_json(json_path, 'label2wnid.json')


class ImagenetExperiment(PytorchNNExperiment):
    """100 natural images with objects analog to the ReberExperiment"""

    def __init__(self):
        super(ImagenetExperiment, self).__init__()
        self.image_path = Path('/home/j.lappalainen/Data/imagenet_stimuli/100')
        self.image_names = self.load_lines(self.image_path, 'img_files.txt')
        self.labels = ['wild_animals',
                       'fruit',
                       'flowers',
                       'insects',
                       'birds',
                       'manmade_food',
                       'clothes',
                       'furniture',
                       'instruments',
                       'computer']
        self.mask = {'total': [None, None, None, None],
                     'within': {'natural': [0, 49, 0, 49],
                                'manmade': [50, 100, 50, 100]},
                     'across': [0, 49, 50, 100]}
        self.num_images_per_label = 10
        self.image_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.num_labels = len(self.labels)
        self.label_abbr = self._abbreviate(self.labels)
        self.label_pretty = self._pretty(self.labels)
        self._mask_keys()
        self.mask_keys_pretty = self._pretty(self.mask_keys)
        self.default_label_colors()
        self.update_leaf_colors()
        self.correlation_colors = {'total': '#4c72b0',
                                   'within': '#c44e52',
                                   'natural': '#55a868',
                                   'manmade': '#8172b2',
                                   'across': '#64b5cd'}
        self.num_images = self._num_images()
        self.mat_file = mat_path / 'zvalsdict.mat'
        self.norm = dict(pytorch=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.norm.update(maxperf=dict(mean=np.ones(3) * 0.45, std=np.ones(3) * 0.15))
        self.norm.update(data=dict(mean=[0.8225, 0.7497, 0.7167], std=[0.2486, 0.2974, 0.3164]))
        self.fig_path = fig_path / 'rdm/imagenetdataset'


class CocoExperiment(Experiment):
    """Images from MSCOCO dataset"""

    def __init__(self):
        super(CocoExperiment, self).__init__()
        self.image_path = ('/home/j.lappalainen/Data/cocoapi/annotations/'
                           'instances_train2014.json')
        self.labels = ['bird',
                       'elephant',
                       # 'bear',
                       # 'zebra',
                       'giraffe',
                       # 'banana',
                       'apple',
                       'orange',
                       # 'broccoli',
                       # 'carrot',
                       # 'sandwich',
                       # 'hot dog',
                       'pizza',
                       'donut',
                       'chair',
                       # 'couch',
                       # 'bed',
                       'tv',
                       # 'laptop',
                       'keyboard'
                       # 'mouse',
                       ]
        self.num_images_per_label = 10
        self.mask = {'total': [None, None, None, None],
                     'within': {'natural': [0, 49, 0, 49],
                                'manmade': [50, 100, 50, 100]},
                     'across': [0, 49, 50, 100]}
        self._mask_keys()
        self.default_correlation_colors()
        self.label_abbr = self._abbreviate(self.labels)
        self.label_pretty = self._pretty(self.labels)
        self.mask_keys_pretty = self._pretty(self.mask_keys)
        self.default_label_colors()
        self.update_leaf_colors()
        self.num_images = self._num_images()
        self.cocodir = Path('/home/j.lappalainen/Data/cocoapi/images')
        self.fig_path = fig_path / 'rdm/mscocodataset'
        self.dataset = {'bird': [265446,
                                 217034,
                                 212359,
                                 512175,
                                 3040,
                                 134644,
                                 117445,
                                 431598,
                                 494675,
                                 135126],
                        'elephant': [168289,
                                     13914,
                                     467727,
                                     528067,
                                     324275,
                                     136002,
                                     212247,
                                     385157,
                                     58385,
                                     332315],
                        'giraffe': [464321,
                                    524369,
                                    526782,
                                    403197,
                                    73591,
                                    284469,
                                    508969,
                                    24990,
                                    110084,
                                    325229],
                        'apple': [58008,
                                  357925,
                                  105697,
                                  331505,
                                  36059,
                                  71784,
                                  308829,
                                  137273,
                                  89892,
                                  517340],
                        'orange': [375066,
                                   134538,
                                   531391,
                                   120491,
                                   411750,
                                   388616,
                                   457703,
                                   219523,
                                   525342,
                                   110257],
                        'pizza': [416408,
                                  52384,
                                  191205,
                                  542642,
                                  189903,
                                  394794,
                                  247789,
                                  21595,
                                  514544,
                                  56486],
                        'donut': [32115,
                                  36413,
                                  298475,
                                  366417,
                                  435620,
                                  12023,
                                  185070,
                                  22080,
                                  147914,
                                  335928],
                        'chair': [149973,
                                  281122,
                                  437537,
                                  143030,
                                  10903,
                                  121841,
                                  170516,
                                  538555,
                                  86192,
                                  362643],
                        'tv': [450330,
                               573973,
                               100303,
                               10476,
                               376397,
                               155496,
                               454915,
                               255568,
                               146850,
                               475743],
                        'keyboard': [386127,
                                     111996,
                                     144349,
                                     245451,
                                     385743,
                                     260399,
                                     127418,
                                     427152,
                                     128074,
                                     167581]}