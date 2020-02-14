from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import math
import pywt


try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

import sys

sys.path.append("..")

from estimator.pose_estimator import PoseEstimator

class SmoothWindow(object):
    def __init__(self, size, threshold):
        self.size = size
        self.window = []
        self.threshold = threshold
        
    def add(self, val):
        if len(self.window) >= self.size:
            self.window = self.window[1:]
        self.window.append(val)
    
    def average(self):
        if len(self.window) > 0:
            return [sum(row[i] for row in self.window)/len(self.window) for i in range(len(self.window[0]))]
        return None
    
    def denoise(self):
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(1, w.dec_len)
        # maxlev = 2 # Override if desired
        threshold = self.threshold # Threshold for filtering
        
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(self.window, 'sym4', level=maxlev)
            
        self.window = pywt.waverec(coeffs, 'sym4')
        new_val = self.window[-1]
        self.window = self.window.tolist()
        return new_val

def lerp(a, b, f):
    return f*b+(1-f)*a;

def rotation_matrix_to_attitude_angles(R) :
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2,0], math.sqrt(R[2, 1]*R[2, 1] + R[2, 2]*R[2, 2]))
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll
    
    #alpha = math.atan2(-R[1, 2], R[2, 2])
    #beta = math.asin(R[2, 0])
    #gamma = math.atan2(-R[1, 0], R[0, 0])
    
    #return alpha, beta, gamma

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return x, y, z

UPPER_THRESHOLD = 5*math.pi/180

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    
    self.flip_idx = opt.flip_idx

    self.previous_R = None
    self.previous_T = None
    
    self.smoothWindow = SmoothWindow(15, 10*math.pi/180)
    self.smoothRotation = SmoothWindow(15, 0.00001)

  def process(self, images, return_time=False):
    height, width = images[0].shape[0:2]
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale

    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    height, width = image.shape[0:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
      
    debugger.add_img(image, img_id='multi_pose')
    
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        
        #TODO: eureka!!!
        
        #head = np.asarray(bbox[5:19])
        #head = bbox[5:15]
        #head.append((bbox[7] + bbox[9])/2)
        #head.append((bbox[8] + bbox[10])/2)
        
        down_range = 5
        up_range = 15
        head = bbox[down_range:up_range]
        head = np.asarray(head)
        
        R, T = pose_estimator.solve_pose(head.reshape([int((up_range-down_range)/2), 2]))
        
        body = bbox[15:19]
        body.extend(bbox[27:31])
        
        body = np.asarray(body)
        bR, bT = pose_estimator.solve_pose(body.reshape([4, 2]), body=True)
        
        lerp_factor = 0.1
        if self.previous_R is not None and self.previous_T is not None:
            new_R = lerp(R, self.previous_R, lerp_factor)
            new_T = lerp(T, self.previous_T, lerp_factor)
            
        else:
            new_R = R
            new_T = T
        #print("nR", new_R)
        self.previous_R = new_R
        self.previous_T = new_T
        rmat = cv2.Rodrigues(R)[0]
        #print(rmat)
        yaw, pitch, roll = rotation_matrix_to_attitude_angles(rmat)
        
        brmat = cv2.Rodrigues(bR)[0]
        byaw, bpitch, broll = rotation_matrix_to_attitude_angles(brmat)
        
        """
        average = self.smoothWindow.average()

        if average is not None:
            if math.fabs(yaw - average[0]) > UPPER_THRESHOLD:
                print("yaw bigger")
                yaw = average[0]
            if math.fabs(pitch - average[1]) > UPPER_THRESHOLD:
                pitch = average[1]
            if math.fabs(roll - average[2]) > UPPER_THRESHOLD:
                roll = average[2]
        
        self.smoothWindow.add([yaw, pitch, roll])
        """
        
        self.smoothWindow.add([yaw, pitch, roll])
        yaw, pitch, roll = self.smoothWindow.denoise()
        
        self.smoothRotation.add(new_R)
        new_R = self.smoothRotation.denoise()
    
        #convert to Unreal coordinate system
        uyaw = -pitch
        upitch = roll
        uroll = yaw
        
        buyaw = -bpitch
        bupitch = broll
        buroll = byaw
        
        
        pose_estimator.evaluation(image, head, new_R, new_T)
        
        pose_estimator.evaluation(image, body, bR, bT, body=True)
        

        cv2.imshow("Preview", image)
        
        debugger.show_all_imgs(pause=self.pause)
        
        position = new_T.reshape([1, 3])[0]
        position = np.array([position[2], -position[0], -position[1]])
        
        bposition = bT.reshape([1, 3])[0]
        bposition = np.array([bposition[2], -bposition[0], -bposition[1]])

        return [upitch, uyaw, uroll], position.tolist(), [bupitch, buyaw, buroll], bposition.tolist(), bbox
    return [None, None, None], None, [None, None, None], None, None