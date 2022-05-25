from __future__ import print_function
import os
import pickle
from re import S

import numpy
from data import get_test_loader, get_transform
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict

import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from data import get_paths
from pycocotools.coco import COCO
from PIL import Image  
   
   
def get_id(index):
    dpath = os.path.join("data/","stair_captions")
    roots, ids = get_paths(dpath,"stair_captions",False)
    ids = ids["test"]
    ann_id = ids[index]
    json = roots["test"]['cap']
    root = roots["test"]['img']
    coco = COCO(json)
    img_id = coco.anns[ann_id]['image_id']
    path = coco.loadImgs(img_id)[0]['file_name']
    url = coco.loadImgs(img_id)[0]['flickr_url']
    im = Image.open(os.path.join(root,path))
    #im.show()

    caption = coco.anns[ann_id]['caption']
    
    print("index",index)
    print(caption)
    print(path)
    print(url)
    print("----------------------")


s = [6289,8800,7880,7240,6365,6175,6570,8495,5125,9120,9325]

count = 0
for i in s:
    if count != 0:
        print("rank:",count)
    get_id(i)
    count += 1