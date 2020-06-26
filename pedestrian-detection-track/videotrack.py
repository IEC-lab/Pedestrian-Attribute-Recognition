from __future__ import print_function
import sys
import os
import pickle
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform,VOCDetection, BaseTransform, VOC_Config
from models.RFB_Net_vgg import build_net
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
import cv2
from torchscope import scope
from compute_speed import compute_speed
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort import preprocessing
import copy

# weights/epoches_022.pth
# Finished loading model!
# 100%|██████████| 3746/3746 [01:35<00:00, 44.04it/s]
# Evaluating detections
# Writing person VOC results file
# VOC07 metric? Yes
# AP for person = 0.6455
# Mean AP = 0.6455
# ~~~~~~~~
# Results:
# 0.646
# 0.646
# ~~~~~~~~
parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('--video_dir', default='video', type=str,
                    help='Dir to save results')
parser.add_argument('-m', '--trained_model', default='weights/646.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()


cfg = VOC_Config
img_dim = 300
num_classes = 2
rgb_means = (104, 117, 123)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=2, thresh=0.1, cuda=True):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.num_classes = num_classes
        self.thresh = thresh
        self.cuda = cuda

    def predict(self, img):
        _t = {'im_detect': Timer(), 'misc': Timer()}
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            #print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = nms(c_dets, 0.4, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        nms_time = _t['misc'].toc()
        total_time = detect_time+nms_time

        #print('total time: ', total_time)
        return all_boxes, total_time

if __name__ == '__main__':
    # load net
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')

    # scope(net, input_size=(3,300,300))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    #scope(net, input_size=(3,300,300))
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    compute_speed(net, (1,3,300,300), device, 1000)
    detector = Detect(num_classes,0,cfg)
    detect_thred=0.32
    transform = BaseTransform(img_dim, rgb_means, (2, 0, 1))
    object_detector = ObjectDetector(net, detector, transform)

    # vdo=cv2.VideoCapture()
    assert os.path.isfile(args.video_dir), "Error: path error"
    rootpath= "/home/lyf/git-repo/RFSong-multidata/"
    videopath = rootpath+args.video_dir
    print(videopath)
    vdo=cv2.VideoCapture(videopath)
    im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = float(vdo.get(cv2.CAP_PROP_FPS))
    print("rate: ", rate)
    assert vdo.isOpened()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, rate, (im_width, im_height))

    # deep_sort 
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'weights/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    startt=time.time()
    fps=0.0
    cnt=1
    while True:
        ret, frame = vdo.read()  
        if ret != True:
            break

    # img_list = os.listdir(args.img_dir)
    # for i, img in enumerate(img_list):
        # st=time.time()
        # img_name = img
        # img = os.path.join(args.img_dir, img)
        image=frame.copy()
        # print(frame.size)
        # image = cv2.imread(frame)
        detect_bboxes, tim = object_detector.predict(image)
        detect_bboxes_copy = copy.deepcopy(detect_bboxes)
        clsid, det_box = detect_bboxes_copy
        det_box = det_box[det_box[:,len(det_box[0])-1]>=detect_thred, :]
        det_box = det_box[:,:len(det_box[0])-1]
        det_box[:, 2:] -= det_box[:, :2]
        # t0=time.time()
        # print(t0-st)
        features = encoder(image,det_box)
        # t1=time.time()
        # print(t1-t0)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(det_box, features)]
        # # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # t2=time.time()
        # print(t2-t1)
        # #do track
        tracker.predict()
        tracker.update(detections)
        indx=1

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            cv2.putText(image, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 1, (255,0,0),2)
            # trackidpath = rootpath+"person/"+str(track.track_id)+"/"
            # if not os.path.exists(trackidpath):
            # 	os.makedirs(trackidpath)
            # bbox = bbox.astype(np.int32)
            # bbox[0] = max(bbox[0],0)
            # bbox[1] = max(bbox[1],0)
            # bbox[2] = min(bbox[2],im_width)
            # bbox[3] = min(bbox[3],im_height)
            # imagecut = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # cv2.imwrite(trackidpath+str(indx)+".jpg", imagecut)
            # indx+=1

        # t3=time.time()
        # # print(t3-t2)
        # for class_id,class_collection in enumerate(detect_bboxes):
        #     if len(class_collection)>0:
        #         for i in range(class_collection.shape[0]):
        #             if class_collection[i,-1]>detect_thred:
        #                 pt = class_collection[i]
        #                 cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)
        #                 # cv2.putText(image, str(class_collection[i,-1]), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # t4=time.time()
        # # print(t4-t3)
        # sec=(time.time()-st)*1000
        # print(sec)
        # # cv2.imwrite('output/' + img_name, image)
        # #cv2.imshow('result',image)
        # #cv2.waitKey()
        # fps  = ( fps + (1./(time.time()-st)) ) / 2
        # print("fps= %f"%(fps))
        # out.write(image)
        cnt+=1

    print(time.time()-startt, cnt)
    print("fps: ", cnt/(time.time()-startt) )
    print("t_per_frame: ", (time.time()-startt)/cnt )
    vdo.release()
    out.release()
