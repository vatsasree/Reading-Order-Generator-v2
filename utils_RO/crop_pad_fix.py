import pandas as pd
import os
import cv2
import numpy as np

import json
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import skimage
import os
import argparse
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
import glob
from tqdm import tqdm
from PIL import Image
import random
from utils import utilities
from shapely.geometry import Polygon
import pyclipper
import datetime
import warnings
warnings.filterwarnings("ignore")
from utils_RO.RO_utils import filter_boxes, calculate_overlap_percentage, is_box_inside
from utils_RO.word_inference_hisam import jsonl_format

########################
# functions for detecting words
def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def detect_text_HiSAM(args, img_cropped, img_source):
    bboxes = []
    lineboxes = [] 
    # args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")
    if args.model_type == 'vit_s' or args.model_type == 'vit_t':
        efficient_hisam = True
    else:
        efficient_hisam = False
    amg = AutoMaskGenerator(hisam, efficient_hisam=efficient_hisam)
    none_num = 0

    if args.eval:
        os.makedirs(args.eval_out_file, exist_ok=True)
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    preds = []
    image_recall_list = {}

    global paragraphs
    # for path in tqdm(args.input, disable=not args.output):
    path = img_source
    print('ppath', path)

    try:
        img_id = os.path.basename(path).split('.')[0]
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = img_id + '.png'
            out_filename = os.path.join(args.output, img_name)
        else:
            assert len(args.input) == 1
            out_filename = args.output
        
        # image = cv2.imread(path)
        image = img_cropped
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
        img_h, img_w = image.shape[:2]

        if args.use_fgmask:
            fgmask_path = os.path.join(args.existing_fgmask_input, img_id+'.png')
            fgmask = skimage.io.imread(fgmask_path)
            amg.set_fgmask(fgmask)

        amg.set_image(image)

        masks, scores, affinity = amg.predict(
            from_low_res=False,
            fg_points_num=args.total_points,
            batch_points_num=args.batch_points,
            score_thresh=0.5,
            nms_thresh=0.5,
        )  # only return word masks here

        if args.eval:
            if masks is None:
                lines = [{'words': [{'text': '', 'vertices': [[0,0],[1,0],[1,1],[0,1]]}], 'text': ''}]
                paragraphs = [{'lines': lines}]
                result = {
                    'image_id': img_id,
                    "paragraphs": paragraphs
                }
                none_num += 1
            else:
                masks = (masks[:, 0, :, :]).astype(np.uint8)  # word masks, (n, h, w)
                lines = []
                line_indices = []
                for index, mask in enumerate(masks):
                    line = {'words': [], 'text': ''}
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for cont in contours:
                        epsilon = 0.002 * cv2.arcLength(cont, True)
                        approx = cv2.approxPolyDP(cont, epsilon, True)
                        points = approx.reshape((-1, 2))
                        if points.shape[0] < 4:
                            continue
                        pts = unclip(points)
                        if len(pts) != 1:
                            continue
                        pts = pts[0].astype(np.int32)
                        if Polygon(pts).area < 32:
                            continue
                        pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
                        pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
                        cnt_list = pts.tolist()
                        xmin = min(v[0] for v in cnt_list)
                        ymin = min(v[1] for v in cnt_list)
                        xmax = max(v[0] for v in cnt_list)
                        ymax = max(v[1] for v in cnt_list)
                        line['words'].append({'text': '', 'vertices': [xmin,ymin,xmax,ymax]})
                    if line['words']:
                        lines.append(line)
                        line_indices.append(index)

                line_grouping = utilities.DisjointSet(len(line_indices))
                affinity = affinity[line_indices][:, line_indices]
                for i1, i2 in zip(*np.where(affinity > args.layout_thresh)):
                    line_grouping.union(i1, i2)
                line_groups = line_grouping.to_group()
                paragraphs = []
                for line_group in line_groups:
                    paragraph = {'lines': []}
                    for id_ in line_group:
                        paragraph['lines'].append(lines[id_])
                    if paragraph:
                        paragraphs.append(paragraph)
                result = {
                    'image_id': img_id,
                    "paragraphs": paragraphs
                }
            # with open(os.path.join(args.eval_out_file, img_id+'.jsonl'), 'w', encoding='utf-8') as fw:
            #     json.dump(result, fw)
            # fw.close()
            
            # lines below added to a different function

            # for paragraph in paragraphs:
            #     for line in paragraph['lines']:
            #         for word in line['words']:
            #             bboxes.append(word['vertices'])
            
            # # bboxes = np.array(bboxes)
            # # print(bboxes)
            # # return bboxes

            # pred = bboxes
            # gt = []
            # irl.append((0, 0, 0, img_id, pred, gt))
            # image_recall_list[iou] = irl
    except Exception as e:
        # continue
        pass
    return paragraphs
########################


########################
# functions for lines
def check_bbox_or_polygon(json_path):
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    data = json_path
    vertices =  data[0]['lines'][0]['words'][0]['vertices']
    if isinstance(vertices, list) and all(isinstance(i, list) for i in vertices):
        return "polygon"
    else:
        return "bbox"

def extract_boxes_from_json_per_line(json_path):
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    data = json_path
    boxes_per_line = []
    for paragraph in data:
        for line in paragraph['lines']:
            words_in_line = []
            for word in line['words']:
                list_coordinate = word['vertices']
                tuple_coordinate = (list_coordinate[0], list_coordinate[1], list_coordinate[2], list_coordinate[3])
                words_in_line.append(tuple_coordinate)
            boxes_per_line.append(words_in_line)
    return boxes_per_line

def single_folder_evaluate_lines(args, paragraphs):
    print('Processing lines.......')
    dict = {}
    dict['lines'] = {}
    json_path = paragraphs

    # image = cv2.imread(image_path)
    bboxes=[]
    if check_bbox_or_polygon(json_path) == "bbox":
        boxes_per_line = extract_boxes_from_json_per_line(json_path)
        for boxes in boxes_per_line:
            min_x = min([box[0] for box in boxes])
            min_y = min([box[1] for box in boxes])
            max_x = max([box[2] for box in boxes])
            max_y = max([box[3] for box in boxes])
            bboxes.append([min_x, min_y,max_x,max_y])
            # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), line_colour, 2)
    dict['lines'] = bboxes
    return dict
########################


########################
# functions for paras
def extract_boxes_from_json_per_para(json_path):
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    data = json_path
    boxes_per_line = []
    for paragraph in data:
        words_in_line = []
        for line in paragraph['lines']:
            for word in line['words']:
                list_coordinate = word['vertices']
                tuple_coordinate = (list_coordinate[0], list_coordinate[1], list_coordinate[2], list_coordinate[3])
                words_in_line.append(tuple_coordinate)
        boxes_per_line.append(words_in_line)
    return boxes_per_line

def single_folder_evaluate_paras(args,paragraphs):
    dict = {}
    dict['paras'] = {}
    json_path = paragraphs

    # image = cv2.imread(image_path)
    bboxes=[]
    if check_bbox_or_polygon(json_path) == "bbox":
        boxes_per_line = extract_boxes_from_json_per_para(json_path)
        for boxes in boxes_per_line:
            min_x = min([box[0] for box in boxes])
            min_y = min([box[1] for box in boxes])
            max_x = max([box[2] for box in boxes])
            max_y = max([box[3] for box in boxes])
            bboxes.append([min_x, min_y,max_x,max_y])
            # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), line_colour, 2)
    dict['paras'] = bboxes
    return dict
########################


########################
#functions for crop pad fix
def save_cropped(word_bboxes, img_dir):

    pad_factor = 0
    img = cv2.cvtColor(cv2.imread(img_dir),cv2.COLOR_BGR2RGB)
    org_TL = (0,0)
    org_BR = (img.shape[1],img.shape[0])

    # preds = doctr_predictions(img_dir)
    preds = word_bboxes
    print('PRED',preds)

    top1=[]
    left1=[]
    bottom1=[]
    right1 = []

    for i in preds:
        left1.append(i[0])
        top1.append(i[1])
        right1.append(i[2])
        bottom1.append(i[3])

    l = min(left1)
    r = max(right1)
    t = min(top1)
    b = max(bottom1)
    # print(l,r,t,b)

    top_left = (l-pad_factor,t-pad_factor)
    bottom_right = (r+pad_factor,b+pad_factor)

    # print('cropped')
    # print(top_left, bottom_right)

    # difference_TL = (top_left[0]-org_TL[0],top_left[1]-org_TL[1])
    # difference_BR = (abs(bottom_right[0]-org_BR[0]),abs(bottom_right[1]-org_BR[1]))
    # print(difference_TL,difference_BR) 
    x1,y1 = top_left
    x2,y2 = bottom_right

    cv2.rectangle(img,top_left, bottom_right,(0,255,255),2)
    # plt.imshow(img1)
    imgg1 = img[y1:y2, x1:x2]
    # plt.imshow(imgg1)
    # cv2.imwrite('/home2/sreevatsa/cropped_image.png',imgg1) #no need of saving cropped image

    return top_left,imgg1

def rescaled_bboxes_from_cropped(args,img_cropped,img_source,top_left, region):
    left = top_left[0]
    top = top_left[1]

    # img_source = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    # target_h = img_source.shape[0]
	# target_w = img_source.shape[1]
    hisam_result = detect_text_HiSAM(args, img_cropped, img_source)
    # print('HISAM_RESULT',hisam_result)
    if region == 'words':
        word_json_nfiltered = jsonl_format(args, hisam_result)
        word_json = filter_boxes(word_json_nfiltered['words'],threshold_percentage=90, type='words')
        region_bbox = word_json['words']
    elif region == 'lines':
        line_bboxes_uf = single_folder_evaluate_lines(args, hisam_result)
        print(line_bboxes_uf)
        line_bboxes = filter_boxes(line_bboxes_uf['lines'],threshold_percentage=90, type='lines')
        region_bbox = line_bboxes['lines']
    elif region == 'paras':
        para_bboxes_uf = single_folder_evaluate_paras(args, hisam_result)
        para_bboxes = filter_boxes(para_bboxes_uf['paras'],threshold_percentage=90, type='paras')
        region_bbox = para_bboxes['paras']

    abs_coords = [[x1+left,y1+top,x2+left,y2+top] for box in region_bbox for x1,y1,x2,y2 in [box]]
    return abs_coords
    # return word_json

def visualized_rescaled_bboxes_from_cropped(args, img_cropped,img_source,top_left, region):
    preds = rescaled_bboxes_from_cropped(args, img_cropped,img_source,top_left, region)
    # img = cv2.cvtColor(cv2.imread(img_source),cv2.COLOR_BGR2RGB)
    # for w in preds[0]:
    # 	cv2.rectangle(img,(w[0], w[1]),(w[2], w[3]),(0,0,255),1)
    # # plt.imshow(img)
    # # cv2.imwrite('/home2/sreevatsa/afterfixoutput_test_doctrv2_{}_{}.png'.format(d,formatted_time), cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    # return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return preds


def cropPadFix(args, word_bboxes, image_file, region):
    # visualize_preds_dir(args.ImageFile)
    cropped_TL, img_cropped = save_cropped(word_bboxes, image_file)
    # visualize_preds_dir('/home2/sreevatsa/cropped.png')
    img = visualized_rescaled_bboxes_from_cropped(args, img_cropped,image_file,cropped_TL, region)
    return {f'{region}':img}

########################
