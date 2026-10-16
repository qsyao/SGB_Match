from .head import *
from .leg import *
from .hand import *
from .chest import *


def select_dataset_SSL_Train(dataset):
    if dataset == 'head':
        return Head_SSL_Train
    elif dataset == 'leg':
        return Leg_SSL_Train
    elif dataset == 'chest':
        return Chest_SSL_Train
    elif dataset == 'hand':
        return Hand_SSL_Train

def select_dataset_SSL_Infer(dataset):
    if dataset == 'head':
        return Head_SSL_Infer
    elif dataset == 'leg':
        return Leg_SSL_Infer
    elif dataset == 'chest':
        return Chest_SSL_Infer
    elif dataset == 'hand':
        return Hand_SSL_Infer

