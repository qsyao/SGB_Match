import torchvision.transforms as transforms
import numpy as np
import os, random, math, torch
import torch.utils.data as data
import json
import pandas as pd
from PIL import Image
import cv2
from copy import deepcopy
from .augment import cc_augment, augment_patch, to_PIL, gray_to_PIL
from utils import radial
from .base import BaseTrainDataset, aug_img_and_kps, BaseInferDataset

WRIST_WIDTH = 50

class Hand_Base(object):
    # Hand radiographs are normalized to 1500x1500 before any CC2D resizing.
    # CSV labels are x,y pairs; physical scale is estimated per image from
    # wrist width instead of fixed DICOM spacing.

    def __init__(self, cfg=None, mode='Train', **kwargs):
    
        self.num_landmark = 37
        self.pathDataset = cfg.dataset.dataset_pth
        self.mode = mode
        self.id_shot = None
        if 'id_shot' in kwargs and self.mode == 'Oneshot':
            self.id_shot = kwargs['id_shot']
            assert self.id_shot is not None
        
    def load_data(self):
        self.pth_Image = os.path.join(self.pathDataset, 'jpg')
        self.labels = pd.read_csv(os.path.join(
            self.pathDataset, 'all.csv'), header=None, index_col=0)

        # file index
        index_set = set(self.labels.index)
        files = sorted(os.listdir(self.pth_Image))
        files = [i for i in files if '_' not in i]
        files = [i[:-4] for i in files]
        files = [i for i in files if int(i) in index_set]
        
        n = len(files)
        train_num = 550  # round(n*0.7)
        val_num = 59  # round(n*0.1)
        test_num = n - train_num - val_num
        if self.mode == 'Train' or self.mode == 'Infer_Train':
            self.indexes = files[:train_num]
        elif self.mode == 'validate':
            self.indexes = files[train_num:-test_num]
        elif self.mode == 'Test':
            self.indexes = files[-test_num:]
        elif self.mode == 'Oneshot':
            self.indexes = files[self.id_shot:self.id_shot+1]
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=self.mode))
        
        self.list = [{'ID': "{}".format(i)} for i in self.indexes]

        self.images = {item['ID']: Image.open(os.path.join(self.pth_Image, item['ID']+'.jpg')).resize([1500,1500]).convert('RGB')\
            for item in self.list}
        
        self.gts = {item['ID']: self._get_landmark_gt(item['ID'])\
            for item in self.list}
        
        self.paths = {item['ID']: os.path.join(self.pth_Image, item['ID']+'.jpg') for item in self.list}
        
        self.scale_rates = {item['ID']: self._compute_spacing_init(self.gts[item['ID']])\
            for item in self.list}

    def __len__(self):
        return len(self.list)

    def readLandmark(self, name):
        li = list(self.labels.loc[int(name), :])
        origin_size = Image.open(os.path.join(self.pth_Image, name+'.jpg')).convert('RGB').size
        r1, r2 = [i/j for i, j in zip([1500,1500], origin_size)]
        points = [list([round(li[i]*r1), round(li[i+1]*r2)])
                  for i in range(0, len(li), 2)]
        return points
    
    def _compute_spacing_init(self, gt):
        # Convert pixel error to mm by assuming the distance between landmarks
        # 0 and 4 corresponds to WRIST_WIDTH millimeters.
        physical_factor = WRIST_WIDTH/radial(gt[0], gt[4])
        return [physical_factor, physical_factor]

    def _get_original_spacing(self, str_ID):
        return self.scale_rates[str_ID]
    
    def _get_landmark_gt(self, name):
        points = self.readLandmark(name)
        return points
    
    def get_landmark_gt(self, name):
        return self._get_landmark_gt(name)
    
    def _get_image(self, item):
        item['image'] = self.basic_transform(self.images[item['ID']])
        return item['image']
            
class Hand_SSL_Train(Hand_Base, BaseTrainDataset):
    def __init__(self, cfg, mode, size=[384, 384], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None, id_oneshot=None):
        Hand_Base.__init__(self, cfg=cfg, mode=mode, pathDataset=cfg.dataset.dataset_pth, size=size, id_shot=id_oneshot,\
                         use_probmap=use_probmap, radius_ratio=radius_ratio, min_prob=min_prob, tag_ssl=tag_ssl)
        BaseTrainDataset.__init__(self, cfg, mode, size=size, use_probmap=use_probmap, radius_ratio=radius_ratio, \
                                  min_prob=min_prob, tag_ssl=tag_ssl)

class Hand_SSL_Infer(Hand_Base, BaseInferDataset):
    def __init__(self, cfg, mode, size=[384, 384], id_oneshot=1):
        Hand_Base.__init__(self, cfg=cfg, size=size, mode=mode, id_shot=id_oneshot)
        BaseInferDataset.__init__(self, cfg=cfg, mode=mode, size=size)

    
