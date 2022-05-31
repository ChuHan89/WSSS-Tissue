#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:02:48 2020

@author: maijinhai
"""

import glob
import numpy as np
import cv2
import imageio
import os

GT_root = '/media/maijinhai/T004/Lung_data/Lung_patch_v2/GT'
# npy_root = '/media/maijinhai/T004/Lung_data/Lung_patch_v2/GT_npy/'
npy_root = '/media/maijinhai/T004/allin/GT_npy/'
save_png_path = '/media/maijinhai/T004/allin/GT_png/'
gt = glob.glob(GT_root+'/*.png')

color_map_gt = [(0,0,164), (0,0,0), (100,9,16), (31,86,0), (160,160,160)] #BGR
for i in gt:
    gt_png = np.uint32(cv2.imread(i))  #BGR
    # gt_name = i.split('/')[-1].split('[')[0][0:-1]
    gt_name = i.split('/')[-1].split('.')[0]
    code = np.zeros((224, 224), dtype="uint8")
    S = gt_png[:,:,0] + gt_png[:,:,1] + gt_png[:,:,2]
    code[S==164] = 0
    code[S==0] = 1
    code[S==125] = 2
    code[S==117] = 3
    code[S==480] = 4
    U = list(np.unique(S))
    
    if not (set(U) <= set([164,0,125,117,480])):
        print('error')
        print(i)
        break
    else:
        print('successfully transform {}'.format(gt_name))
        
    
    np.save(npy_root + gt_name + '.npy', code)
    # imageio.imwrite(os.path.join(save_png_path, gt_name)+'.jpg', code)
    
# gt = np.uint32(cv2.imread(GT_root + '/387709-9108-71176-[1 0 0 1].png'))
# gt[:,:,0]+gt[:,:,1]+gt[:,:,2]==480
# gt_png = np.uint32(cv2.imread('/media/maijinhai/T004/Lung_data/Lung_patch_v2/GT/416003-26126-48014-[0 0 0 1].png'))
# np.where(S==118)
