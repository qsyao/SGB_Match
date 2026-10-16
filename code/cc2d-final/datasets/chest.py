import torchvision.transforms as transforms
import numpy as np
import os, random, math, torch
import torch.utils.data as data
import json
from PIL import Image
import cv2
from copy import deepcopy
from .augment import cc_augment, augment_patch, to_PIL, gray_to_PIL
from .base import BaseTrainDataset, aug_img_and_kps, BaseInferDataset

class Chest_Base(object):
    # Chest labels are stored as normalized ratios. Images and labels are first
    # mapped to 1024x1024, then BaseDataset handles the coarse 384x384 resize.
    
    def __init__(self, cfg=None, mode='Train', **kwargs):
        pathDataset = cfg.dataset.dataset_pth
        self.num_landmark = 6
        self.mode = mode
        self.pixel_spaceing = 1

        self.pth_Image = os.path.join(pathDataset, 'pngs')
        self.pth_Label = os.path.join(pathDataset, 'labels')

        self.id_shot = None
        if 'id_shot' in kwargs and self.mode == 'Oneshot':
            self.id_shot = kwargs['id_shot']
            assert self.id_shot is not None

    def load_data(self):
        # file index
        files = sorted(os.listdir(self.pth_Image))
        files = [i for i in files if '_edge' not in i]
        files = [i for i in files if '_sobel' not in i]
        files = files[:279]
        files = [i[:-4] for i in files]
        
        n = len(files)
        train_num = 195  # round(n*0.7) # 180
        val_num = 34  # round(n*0.1)  # 24
        test_num = n - train_num - val_num
        if self.mode == 'Train' or self.mode == 'Infer_Train':
            self.indexes = files[:train_num]
        elif self.mode == 'Validate':
            self.indexes = files[train_num:-test_num]
        elif self.mode == 'Test':
            self.indexes = files[-test_num:]
        elif self.mode == 'Oneshot':
            self.indexes = files[self.id_shot:self.id_shot+1]
        else:
            raise Exception(f"Unknown mode: {self.mode}")
        
        self.list = [{'ID': "{}".format(i)} for i in self.indexes]

        self.images = {item['ID']: Image.open(os.path.join(self.pth_Image, item['ID']+'.png')).convert('RGB').resize((1024, 1024))\
            for item in self.list}
        
        self.paths = {item['ID']: os.path.join(self.pth_Image, item['ID']+'.png') for item in self.list}

        self.gts = {item['ID']: self.get_landmark_gt(item['ID']) for item in self.list}

        # Evaluation uses a fixed 0.5 mm/pixel scale after 1024x1024 normalization.
        self.scale_rates = {item['ID']: [0.5, 0.5] for item in self.list}
    
    def __len__(self):
        return len(self.list)

    def get_landmark_gt(self, name):
        path = os.path.join(self.pth_Label, name+'.txt')
        points = []
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                ratios = [float(i) for i in f.readline().split()]
                pt = tuple([round(r*sz) for r, sz in zip(ratios, [1024, 1024])])
                points.append([pt[0], pt[1]])
        return points


    def _get_image(self, item):
        img = Image.open(os.path.join(self.pth_Image, item['ID']+'.png')).convert('RGB')
        return self.basic_transform(img)

    def _get_original_spacing(self, str_ID):
        return self.pixel_spaceing, self.pixel_spaceing



class Chest_SSL_Train(Chest_Base, BaseTrainDataset):
    def __init__(self, cfg, mode, size=[384, 384], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None, id_oneshot=None):
        Chest_Base.__init__(self, cfg=cfg, mode=mode, pathDataset=cfg.dataset.dataset_pth, size=size, id_shot=id_oneshot,\
                         use_probmap=use_probmap, radius_ratio=radius_ratio, min_prob=min_prob, tag_ssl=tag_ssl)
        BaseTrainDataset.__init__(self, cfg, mode, size=size, use_probmap=use_probmap, radius_ratio=radius_ratio, \
                                  min_prob=min_prob, tag_ssl=tag_ssl)

class Chest_SSL_Infer(Chest_Base, BaseInferDataset):
    def __init__(self, cfg, mode, size=[384, 384], id_oneshot=1):
        Chest_Base.__init__(self, cfg=cfg, size=size, mode=mode, id_shot=id_oneshot)
        BaseInferDataset.__init__(self, cfg=cfg, mode=mode, size=size)