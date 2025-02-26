import cv2
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import json
import tqdm
import argparse

def get_yolo_layout(args):
    model = YOLO(args.yolo_checkpoint)
    model = model.to(args.device)
    
    dict = {}
    imglist = args.input
    for i,img_name in tqdm.tqdm(enumerate(imglist)):
        try:
            source = cv2.imread(img_name)

            results = model([source])  # list of Results objects
            names = results[0].names
            # print(names)

            for r in results:
                temp_dict={}
                classes = r.boxes.cls
                xyxy = r.boxes.xyxy
                # print(classes)
                print(xyxy)
                for i in range(len(classes)):
                    # print(int(classes[i]))
                    if names[int(classes[i])] in temp_dict:
                        temp_dict[names[int(classes[i])]].append(xyxy[i].tolist())
                    else:
                        temp_dict[names[int(classes[i])]] = [xyxy[i].tolist()]
                # dict[img_name.split('/')[-1].split('.')[0]] = temp_dict
                dict['layout'] = temp_dict
        except Exception as e:
            print(f'Error in {img_name} : {e}')
            continue
    # print(dict)
    
    return dict