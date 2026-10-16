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


class Leg_Base(object):
    # Leg images are highly rectangular. PIL resize uses (width, height), so
    # coarse_size below is [W, H] = [2400, 5800]. BaseDataset then converts
    # x,y landmarks into [y, x] for model training/evaluation.
    def __init__(self, cfg=None, mode='Train', **kwargs):
        self.pathDataset = cfg.dataset.dataset_pth
        self.num_landmark = 20
        self.list = []
        self.pathDataset = cfg.dataset.dataset_pth
        self.mode = mode
        self.id_shot = None
        if 'id_shot' in kwargs and self.mode == 'Oneshot':
            self.id_shot = kwargs['id_shot']
            assert self.id_shot is not None
    
    def load_data(self):
        if self.mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pathDataset, 'train')
            start = self.id_shot
            self.list.append({'ID': "{}".format(start)})
        elif self.mode == 'Train' or self.mode == 'Infer_Train':
            self.pth_Image = os.path.join(self.pathDataset, 'train')
            start = 1
            end = 120
            for i in range(start, end + 1):
                self.list.append({'ID': "{}".format(i)})
        elif self.mode == 'Test1':
            self.pth_Image = os.path.join(self.pathDataset, 'test1')
            with open(os.path.join(self.pathDataset, 'test1.json'), 'r') as f:
                self.list = [{'ID': "{}".format(i)} for i in json.load(f)]
        elif self.mode == 'Test2':
            self.pth_Image = os.path.join(self.pathDataset, 'test2')
            with open(os.path.join(self.pathDataset, 'test2.json'), 'r') as f:
                self.list = [{'ID': "{}".format(i)} for i in json.load(f)]
        else:
            self.pth_Image = os.path.join(self.pathDataset, 'test')
            with open(os.path.join(self.pathDataset, 'test.json'), 'r') as f:
                self.list = [{'ID': "{}".format(i)} for i in json.load(f)]
        
        self.coarse_size = [2400, 5800]
        self.images = {item['ID']: Image.open(os.path.join(self.pth_Image, item['ID']+'.png')).convert('RGB').resize(self.coarse_size)\
            for item in self.list}
        
        self.paths = {item['ID']: os.path.join(self.pth_Image, item['ID']+'.png') for item in self.list}

        self.gts = {item['ID']: self.get_landmark_gt(item['ID']) for item in self.list}

        self.scale_rates = {item['ID']: self._get_spacing(item['ID']) for item in self.list}
        

    def get_landmark_gt(self, id_str, get_scale_rate=False):
        with open(os.path.join(self.pth_Image, f'{id_str}_meta.json')) as f:
            meta_info = json.load(f)

        landmark_list = meta_info['gt']
        del landmark_list[23:]
        del landmark_list[10:13]
        
        original_shape = meta_info['original_shape']
        # Meta landmarks are x,y in original pixels. Scale each axis into the
        # 2400x5800 normalized image before BaseDataset applies coarse/fine logic.
        landmark_ori = deepcopy(landmark_list)
        for id, landmark in enumerate(landmark_list):
            landmark_list[id] = [landmark[0] * self.coarse_size[0] / original_shape[1], landmark[1] * self.coarse_size[1] / original_shape[0]]
        
        return landmark_list
        
    def _get_spacing(self, id_str):
        # Scale physical spacing by the same resize factors used for the image,
        # so radial error is reported in millimeters after normalization.
        with open(os.path.join(self.pth_Image, f'{id_str}_meta.json')) as f:
            meta_info = json.load(f)
            meta_info['spacing'] = [meta_info['spacing'][0] * meta_info['original_shape'][1] / self.coarse_size[0], \
                                    meta_info['spacing'][1] * meta_info['original_shape'][0] / self.coarse_size[1]]
        return meta_info['spacing']

class Leg_SSL_Train(Leg_Base, BaseTrainDataset):
    def __init__(self, cfg, mode, size=[384, 384], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None, id_oneshot=None):
        Leg_Base.__init__(self, cfg=cfg, mode=mode, pathDataset=cfg.dataset.dataset_pth, size=size, id_shot=id_oneshot,\
                         use_probmap=use_probmap, radius_ratio=radius_ratio, min_prob=min_prob, tag_ssl=tag_ssl)
        BaseTrainDataset.__init__(self, cfg, mode, size=size, use_probmap=use_probmap, radius_ratio=radius_ratio, \
                                  min_prob=min_prob, tag_ssl=tag_ssl)

class Leg_SSL_Infer(Leg_Base, BaseInferDataset):
    def __init__(self, cfg, mode, size=[384, 384], id_oneshot=1):
        Leg_Base.__init__(self, cfg=cfg, size=size, mode=mode, id_shot=id_oneshot)
        BaseInferDataset.__init__(self, cfg=cfg, mode=mode, size=size)

if __name__ == "__main__":
    # hamming_set(9, 100)
    # test = Test_Cephalometric('../../dataset/Cephalometric', 'Oneshot')
    # for i in range(150):
    #     test.__getitem__(i)
    # test = Leg_SSL_Train('../../dataset/Cephalometric', 'Train', use_probmap=True, radius_ratio=0.05, min_prob=0.1)
    test = Leg_SSL_Infer('../../dataset/Cephalometric', 'Test')
    for i in range(150):
        test.__getitem__(i)
    print("pass")