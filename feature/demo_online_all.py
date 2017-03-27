#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
import copy
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}
    

def demo(net, image_name, bbox, obj_no, sf_no, iteration):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds

    mFeat = im_detect(net, im, bbox)
    cmFeat = copy.deepcopy(mFeat)

    mask_x1 = int(max(0,bbox[0,0]-(bbox[0,2]-bbox[0,0])/2))
    mask_x2 = int(min(640,bbox[0,2]+(bbox[0,2]-bbox[0,0])/2))
    mask_y1 = int(max(0,bbox[0,1]-(bbox[0,3]-bbox[0,1])/2))
    mask_y2 = int(min(360,bbox[0,3]+(bbox[0,3]-bbox[0,1])/2))

    im[0:mask_x1,:,:]=0
    im[:,0:mask_y1,:]=0
    im[mask_x2:,:,:]=0
    im[:,mask_y2:,:]=0

    bbox_mask = np.zeros((1,4))
    bbox_mask[0,0] = 0
    bbox_mask[0,1] = 0
    bbox_mask[0,2] = 640
    bbox_mask[0,3] = 360

    feat2 = im_detect(net, im, bbox_mask)

    feat_combine = np.concatenate((cmFeat,feat2),axis=1)
    feat_combine = np.reshape(feat_combine,(1,-1))

    filename = 'clipFeat_online' +str('%02d' %iteration)+'/clipFeat_obj_' + str('%03d' %obj_no) +'_sf_' + str('%03d' %sf_no) +'.txt'
    f = open(filename,'a')
    np.savetxt(f,feat_combine)
    f.close()

    # sio.savemat(filename, {'feat':feat_combine}) 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    path_frames = '/home/yjzhang/py-faster-rcnn/data/online_img_all/'

    for iteration in range(1,30+1):
        objclip_name = '/home/yjzhang/fast-rcnn_Feature/obj_clips/objclip_' + str(iteration) +'.mat'
        data = sio.loadmat(objclip_name)
        obj_total_num = int(max(data['object_clip'][:,0]))
        fr_total_num = int(max(data['object_clip'][:,1]))

        for obj_no in range(1,obj_total_num+1):
            if(sum(data['object_clip'][:,0]==obj_no)==0):
                continue
            else:
               clip_current = (data['object_clip'][data['object_clip'][:,0]==obj_no])
               sf_num = int(max(clip_current[:,6]))
               for sf_no in range(1,sf_num+1):
                  sf_current = clip_current[clip_current[:,6]==sf_no]
                  fr_current_num = int(len(sf_current[:,1]))
                  fr_current_start = int(sf_current[0,1])
                  fr_current_end = int(sf_current[fr_current_num-1,1])
                  fr_current_mid = int(sf_current[int(fr_current_num/2),1])

                  for fr_current_no in [fr_current_start,fr_current_mid,fr_current_end]:
                        if fr_current_no > 6000:   
                            frame = path_frames + str('%d' %fr_current_no) + '.jpg'
                        else:
                            frame = path_frames + str('%06d' %fr_current_no) + '.jpg'
 
                        bbox = sf_current[sf_current[:,1]==fr_current_no][0,2:6]
                        bbox_new = np.zeros((1,4)) #[x1,y1,w1,h1]
                        bbox_new[0,0] = max(0,bbox[0])
                        bbox_new[0,1] = max(0,bbox[1])
                        bbox_new[0,2] = min(640,bbox[0]+bbox[2])
                        bbox_new[0,3] = min(360,bbox[1]+bbox[3])
                    
                        demo(net, frame, bbox_new, obj_no, sf_no, iteration)
