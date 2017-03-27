#!/usr/bin/env python

# --------------------------------------------------------
# obtain boundingboxes and scores for image sequences
# using 5 classes:
# 'bicycle', 'bus', 'car', 'motorbike', 'person'
# offline: set conf_thresh = 0.5
# online: set conf_thresh = 0.2
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
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
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print "~~~~~~~~~~~~~~~~~~~~~~~~~"

        filename='bbox_'+str(image_name[:-4])+'.txt'
        f=open(filename,'a')
        np.savetxt(f, bbox)
        f.close()

        filename2='score_'+str(image_name[:-4])+'.txt'
        f2=open(filename2,'a')
        f2.write(str(score))
        f2.write('\n')
        f2.close()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('/home/yjzhang/py-faster-rcnn/data', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5 #offline, online:0.2
    NMS_THRESH = 0.3
    # 
    arr = np.array([2,6,7,14,15],dtype=int)

    # for cls_ind, cls in enumerate(CLASSES[1:]):
    for cls_ind in [2,6,7,14,15]: # classes of moving objects
        # cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, CLASSES[cls_ind], dets, image_name, thresh=CONF_THRESH)
		# vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
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
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    path = "/home/yjzhang/py-faster-rcnn/data/"
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            frame = file
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/{}'.format(file)
            demo(net, frame)
