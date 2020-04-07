from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import csv
import time

from opts import opts
from detectors.detector_factory import detector_factory
from lib.utils.debugger import Debugger

from twisted.internet.defer import inlineCallbacks
from autobahn.websocket.util import parse_url as parseWsUrl
from autobahn.twisted import wamp, websocket
from autobahn.wamp import types
from autobahn.twisted.util import sleep

from datetime import datetime

from queue import Queue
import threading
from threading import Thread, Lock

from twisted.internet import reactor
from twisted.internet.protocol import ReconnectingClientFactory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

outputQueue = Queue(maxsize=2)
broadcasting = False
lock = Lock()

class MyFrontendComponent(wamp.ApplicationSession):
    
    @inlineCallbacks
    def onJoin(self, details):
        print("onJoin")  
        global broadcasting

            
        while broadcasting:
            try:
                content = outputQueue.get()

                self.publish('SynOpticon.OpenFaceSample', "CenterNet", [0] * 16, content[0])
                self.publish('SynOpticon.CenterNetSample.Body', "CenterNet", content[0])
                
                yield sleep(0.08)
            except Exception as e:
                print("exception", e)
                pass

        yield self.leave()

class MyClientFactory(websocket.WampWebSocketClientFactory, ReconnectingClientFactory):
    maxDelay = 30

    def clientConnectionFailed(self, connector, reason):
        print("*************************************")
        print("Connection Failed")
        print("reason:", reason)
        print("*************************************")
        ReconnectingClientFactory.clientConnectionFailed(self, connector, reason)

    def clientConnectionLost(self, connector, reason):
        print("*************************************")
        print("Connection Lost")
        print("reason:", reason)
        print("*************************************")
        ReconnectingClientFactory.clientConnectionLost(self, connector, reason)
        
def saveMetaData(file_name, webcam=True):
    data = {}
    
    meta_file = os.path.join("./output", file_name.split('.')[0] + '.json')
    
    if webcam:
        created_time = time.time()
    else:
        created_time = os.path.getmtime(file_name)
        
    data['created time'] = datetime.utcfromtimestamp(created_time).strftime('%d-%m-%Y %H:%M:%S.%f')
    with open(meta_file, 'w') as outfile:
        json.dump(data, outfile)

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  video_output = None
  
  print("FKKKKKK", opt.demo)
  
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        
    cam = None
    if opt.demo == 'webcam':
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(opt.demo)
        
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    csv_bf = None
    csv_wr = None
    
    if opt.save != '':
        if not os.path.isdir("./output"):
          os.mkdir("./output")
        
        file_name = opt.demo
        
        csv_file_name = opt.demo.split('.')[0] + '.csv'
        if opt.demo == 'webcam':
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S %d_%m_%y")
            file_name = current_time + '.mp4'
            csv_file_name = current_time + '.csv'
            
            saveMetaData(current_time, True)
        else:
            saveMetaData(opt.demo, False)
      
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_output = cv2.VideoWriter(os.path.join("./output", file_name), fourcc, 20.0, (width, height))
        
        csv_bf = open(os.path.join("./output", csv_file_name), 'w', newline='')
        csv_wr = csv.writer(csv_bf, delimiter=',')
        
        
    debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                        theme=opt.debugger_theme, out=video_output)
    
    detector.pause = False
    
    count = 0
    
    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)
       
        orientation, position, borientation, bpos, keypoints = detector.run(img, debugger)
        if opt.wamp != '':
            if orientation is not None and position is not None:
                outputQueue.put((["CenterNet", position, orientation, []], ["CenterNet.Body", position, borientation, []])) 
        
        if csv_wr is not None:
            output_list = [[count], orientation, position, borientation, bpos, keypoints]
            #print("output", output_list)
            line = [item for sublist in output_list for item in sublist]
            csv_wr.writerow(line)
        

        if cv2.waitKey(1) == 113:
            cam.release()
            if video_output is not None:
                video_output.release()
            cv2.destroyAllWindows()
            if csv_bf is not None:
                csv_bf.close()
            
            return  # esc to quit
        
        count += 1
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:   
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
  
if __name__ == '__main__':
  opt = opts().init()

  if opt.wamp != '':
      component_config = types.ComponentConfig(realm=opt.realm)
      session_factory = wamp.ApplicationSessionFactory(config=component_config)
      session_factory.session = MyFrontendComponent
    
      ## 2) create a WAMP-over-WebSocket transport client factory
      url = u"ws://" + opt.crossbar + "/ws"
      transport_factory = MyClientFactory(session_factory, url=url)
    
      ## 3) start the client from a Twisted endpoint
      isSecure, host, port, resource, path, params = parseWsUrl(url)
      transport_factory.host = host
      transport_factory.port = port
      websocket.connectWS(transport_factory)
    
      ## 4) now enter the Twisted reactor loop
      broadcasting = True
      wampThread = Thread(target=reactor.run, args=(False,))
      wampThread.start()
  demo(opt)
  
  if opt.wamp != '':
      lock.acquire()
      broadcasting = False
      lock.release()
      outputQueue.empty()
      reactor.stop()
      wampThread.join(10)
      if wampThread.is_alive():
          print("the asshole is still alive, have no idea why")
