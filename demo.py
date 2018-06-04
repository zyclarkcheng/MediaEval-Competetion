import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2#for ckpt.meta
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
sys.path.insert(0, this_dir )
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

#CLASSES=('__background__','Pedestrain','Car','Cyclist')

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # init session
    demo_net='VGGnet_test'
    model=this_dir+'/VGGnet_fast_rcnn_iter_70000.ckpt'
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    # load network
    net = get_network(demo_net)
    # load model
    print ('Loading network {:s}... '.format(demo_net)),

    saver = tf.train.Saver()

    saver.restore(sess, model)

    print (' done.')
#    f=open(this_dir+'/v0_v25/detect_v0_v25.txt','w')
#    f=open(this_dir+'/v26_v51/detect_v26_v51.txt','w')
    f=open(this_dir+'/v52_v77/detect_v52_v77.txt','w')
#    f=open(this_dir+'/v78_v107/detect_v78_v107.txt','w')

    for cate in range(52,78):
        im_names =  glob.glob(os.path.join(this_dir+'/v52_v77/videos/video_'+str(cate)+'/images/*.jpg'))             
        for im_name in im_names:
            k=0
            im = cv2.imread(im_name)

    # Detect all object classes and regress object bounds
            timer = Timer()
            timer.tic()
            scores, boxes = im_detect(sess, net, im)
            timer.toc()
            print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])


            CONF_THRESH = 0.7
            NMS_THRESH = 0.1
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                class_name=cls
                thresh=0.5
#                vis_detections(im, class_name, dets, ax, thresh=0.5):"""Draw detected bounding boxes."""
                inds = np.where(dets[:, -1] >= thresh)[0]
                if len(inds) == 0:
                    k=k+1

                else:

                    f.write('video_'+str(cate)+','+im_name.split('/')[-1]+','+class_name+','+str(len(inds))+',')                
                    for i in inds:
                        bbox = dets[i, :4]
                        score = dets[i, -1]

                        print class_name,score
                        f.write(str(score)+',')
                    f.write('\n')
            if k==20:
                f.write('video_'+str(cate)+','+im_name.split('/')[-1]+',nothing')
                f.write('\n')


f.close()    
#    plt.show()


