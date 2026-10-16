import torchvision.transforms as transforms
import numpy as np
import os, random, math, torch
import torch.utils.data as data
import json
import cv2
from PIL import Image
from .augment import cc_augment, augment_patch, np_gray_to_PIL, gray_to_PIL, to_PIL
from .base import BaseTrainDataset, aug_img_and_kps, BaseInferDataset

from imgaug.augmentables import KeypointsOnImage

import numpy as np
import imgaug.augmenters as iaa
from skimage import io
import random




class Head_Base(object):
    # Cephalometric images are stored at their native 2400x1935-ish size.
    # Label txt files store x,y pairs from junior/senior annotators; BaseDataset
    # later converts them to the repository-wide [y, x] convention when resizing.
    # the parameters in init should have the same order with BaseTrainDataset
    # cfg, mode, and **kwargs
    def __init__(self, cfg=None, mode='Train', **kwargs):
        self.pathDataset = cfg.dataset.dataset_pth
        self.mode = mode
        self.id_shot = None
        if 'id_shot' in kwargs and self.mode == 'Oneshot':
            self.id_shot = kwargs['id_shot']
            assert self.id_shot is not None
        
    def load_data(self):
    
        self.list = list()
        self.num_landmark = 19

        # ISBI cephalometric pixel spacing is 0.1 mm in both axes.
        self.pixel_spaceing = 0.1

        self.pth_Image = os.path.join(self.pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(self.pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(self.pathDataset, '400_senior')

        if self.mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            assert self.id_shot is not None
            start = self.id_shot
            end = self.id_shot
        elif self.mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif self.mode == 'Infer_Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif self.mode == 'Test':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
            
        # all datasets need a self.list, which is a dictionary with key 'ID'
        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.images = {item['ID']: Image.open(os.path.join(self.pth_Image, item['ID']+'.bmp')).convert('RGB')\
            for item in self.list}

        self.paths = {item['ID']: os.path.join(self.pth_Image, item['ID']+'.bmp') for item in self.list}

        self.gts = {item['ID']: self._get_landmark_gt(item['ID']) for item in self.list}

        self.scale_rates = {item['ID']: self._get_original_spacing(item['ID']) for item in self.list}
    

    def _get_landmark_gt(self, id_str, id_landmark=-1):
        landmark_list = list()

        with open(os.path.join(self.pth_label_junior, id_str+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, id_str+'.txt')) as f2:
                for i in range(19):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    # landmark_list.append(self.resize_landmark_dataset(landmark))
                    landmark_list.append(landmark)

        # if id_landmark is not None and id_landmark >= 0:
        #     landmark_list = [landmark_list[id_landmark]]
        
        return landmark_list
    
    def _get_image(self, item):
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        return self.basic_transform(Image.open(pth_img).convert('RGB'))

    def _get_original_spacing(self, str_ID):
        return self.pixel_spaceing, self.pixel_spaceing


class Head_SSL_Train(Head_Base, BaseTrainDataset):
    def __init__(self, cfg, mode, size=[384, 384], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None, id_oneshot=None):
        Head_Base.__init__(self, cfg=cfg, mode=mode, pathDataset=cfg.dataset.dataset_pth, size=size, id_shot=id_oneshot,\
                         use_probmap=use_probmap, radius_ratio=radius_ratio, min_prob=min_prob, tag_ssl=tag_ssl)
        BaseTrainDataset.__init__(self, cfg, mode, size=size, use_probmap=use_probmap, radius_ratio=radius_ratio, \
                                  min_prob=min_prob, tag_ssl=tag_ssl)
    

class Head_SSL_Infer(Head_Base, BaseInferDataset):
    def __init__(self, cfg, mode, size=[384, 384], id_oneshot=1):
        Head_Base.__init__(self, cfg=cfg, size=size, mode=mode, id_shot=id_oneshot)
        BaseInferDataset.__init__(self, cfg=cfg, mode=mode, size=size)


# if __name__ == "__main__":
#     # hamming_set(9, 100)
#     # test = Test_Cephalometric('../../dataset/Cephalometric', 'Oneshot')
#     # for i in range(150):
#     #     test.__getitem__(i)
#     # test = Head_SSL_Train('../../dataset/Cephalometric', 'Train', use_probmap=True, radius_ratio=0.05, min_prob=0.1)
#     test = Head_SSL_Train('../../dataset/Cephalometric', 'Test')
#     for i in range(150):
#         test.__getitem__(i)
#     print("pass")