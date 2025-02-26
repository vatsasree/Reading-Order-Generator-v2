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

from utils_RO.word_inference_hisam import *
from utils_RO.lines_processing import *
from utils_RO.paras_processing import *
from utils_RO.yolo_layout import *
from utils_RO.RO_utils import *
from utils_RO.crop_pad_fix import cropPadFix

def get_args_parser():
    parser = argparse.ArgumentParser('Hi-SAM', add_help=False)

    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default='./demo',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--output_folder", type=str, default='/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/tests',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--output_folder_images", type=str, default='/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/tests',
                        help="A file or directory to save output visualizations.")
    parser.add_argument("--existing_fgmask_input", type=str, default='./datasets/HierText/val_fgmask/',
                        help="A file or directory of foreground masks.")
    parser.add_argument("--model-type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b', 'vit_s']")
    parser.add_argument("--checkpoint", type=str, required=False, default="/ssd_scratch/sreevatsa/HiSAM/pretrained_checkpoint/hi_sam_l.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--yolo_checkpoint", type=str, required=False, default="/ssd_scratch/sreevatsa/Reading-Order-Generator-v2/best.pt",
                        help="Yolo layout model checkpoint to detect distractor regions.")
    parser.add_argument("--device", type=str, default="cuda:2",
                        help="The device to run generation on.")
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--use_fgmask", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--eval", action='store_true', default=True)
    parser.add_argument("--eval_out_file", type=str, default='./hiertext_eval/consortium_result_jsons',
                        help="A file or directory to save results per image.")
    parser.add_argument('--total_points', default=1500, type=int, help='The number of foreground points')
    parser.add_argument('--batch_points', default=100, type=int, help='The number of points per batch')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)

    # self-prompting
    parser.add_argument('--attn_layers', default=1, type=int,
                        help='The number of image to token cross attention layers in model_aligner')
    parser.add_argument('--prompt_len', default=12, type=int, help='The number of prompt token')
    parser.add_argument('--layout_thresh', type=float, default=0.5)
    parser.add_argument('--save_bboxes_images', default=False, type=bool)

    return parser.parse_args()

args = get_args_parser()

def draw_bboxes(dic,img):
    for box in dic:
        print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2],box[3]), (0,255,0),2 )

def Reading_Order_Generator(args):
    img = cv2.imread((args.input)[0])

    output_folder = args.output_folder
    output_folder_images = args.output_folder_images

    file_name = (args.input)[0].split('/')[-1].split('.')[0]
    reading_order_final = {file_name: {}}

    reading_order_final[file_name] = {}

    hisam_result = detect_text_HiSAM(args)

    word_json_unfiltered = jsonl_format(args, hisam_result)
    word_json = cropPadFix(args, word_json_unfiltered['words'], (args.input)[0], region='words')
    word_json = filter_boxes(word_json['words'],threshold_percentage=90, type='words')
    
    # order = 0
    for i,line in enumerate(word_json['words']):
        # for line in para:
        cv2.rectangle(img, (line[0],line[1]), (line[2],line[3]),(255,0,0),2)
        # cv2.putText(img, str(order),(line[0],line[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
        # order+=1

    # # print("num of words before filtering:", len(word_json_nfiltered['words']))
    # # print("num of words after filtering:", len(word_json['words']))
    # reading_order_final[file_name]['words'] = word_json

    # line_bboxes_uf = single_folder_evaluate_lines(args, hisam_result)
    # # line_bboxes = cropPadFix(args, line_bboxes_uf['lines'], (args.input)[0], region='lines')
    # line_bboxes = filter_boxes(line_bboxes_uf['lines'], threshold_percentage=75, type='lines')

    
    # # for i,line in enumerate(line_bboxes['lines']):
    # #     # for line in para:
    # #     cv2.rectangle(img, (line[0],line[1]), (line[2],line[3]),(0,255,0),2)
    # #     # cv2.putText(img, str(order),(line[0],line[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
    # #     # order+=1

    # # print("num of words before filtering:", len(line_bboxes_uf['lines']))
    # # print("num of words after filtering:", len(line_bboxes['lines']))

    # # print(line_bboxes)
    # reading_order_final[file_name]['lines'] = line_bboxes

    # para_bboxes_uf = single_folder_evaluate_paras(args, hisam_result)
    # # para_bboxes = cropPadFix(args, para_bboxes_uf['paras'], (args.input)[0], region='paras')
    # para_bboxes = filter_boxes(para_bboxes_uf['paras'], threshold_percentage=90, type='paras')

    # # for i,line in enumerate(para_bboxes['paras']):
    # #     # for line in para:
    # #     cv2.rectangle(img, (line[0],line[1]), (line[2],line[3]),(0,0,255),2)
    # #     # cv2.putText(img, str(order),(line[0],line[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
    # #     # order+=1


    # # print("num of words before filtering:", len(para_bboxes_uf['paras']))
    # # print("num of words after filtering:", len(para_bboxes['paras']))

    # reading_order_final[file_name]['paras'] = para_bboxes

    # layout_bboxes_yolo = get_yolo_layout(args)
    # # print(layout_bboxes_yolo)
    # reading_order_final[file_name]['layout'] = layout_bboxes_yolo
    # # for k,v in layout_bboxes_yolo['layout'].items():
    # #     for i in v:
    # #         cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
    # #         cv2.putText(img, k, (int(i[0]), int(i[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # #draw paragraph boxes on images
    
    # # for key, values in layout_bboxes_yolo['layout'].items():
    # #     for value in values:
    # #         cv2.rectangle(img, (int(value[0]), int(value[1])), (int(value[2]), int(value[3])), (0, 0, 255), 1)
    # #         cv2.putText(img, key, (int(value[0]), int(value[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # filtered_paras_layout = filter_paras_layout(para_bboxes, layout_bboxes_yolo)
    # reading_order_final[file_name]['filtered_paras'] = filtered_paras_layout
    # # print(filtered_paras_layout)
    # # filtered_paras = filtered_paras_layout['paras']
    # # for i in filtered_paras:
    # #     # print(i)
    # #     cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
 
    # sorted_paras = get_paragraph_order(filtered_paras_layout['paras'],(args.input)[0],'test.png')
    # # print(sorted_paras)
    # reading_order_final[file_name]['sorted_paras'] = sorted_paras

    # # order = 0
    # # for idx, line in enumerate(sorted_paras):
    # #     for para in line:
    # #         # x1,y1,x2,y2 = para[0],para[1],para[2],para[3]
    # #         cv2.rectangle(img, (para[0], para[1]), (para[2], para[3]), (0, 0, 255), 2)  # Blue box with thickness 2
    # #         cv2.putText(
    # #             img,
    # #             str(order),
    # #             (para[0]-10, para[1] - 10),  # Position the text slightly above the top-left corner
    # #             cv2.FONT_HERSHEY_SIMPLEX,
    # #             0.6,  # Font scale
    # #             (0, 0, 255),  # Blue text
    # #             2  # Thickness
    # #         )
    # #         order+=1

    # lines_sorted_per_para = get_line_order(line_bboxes, sorted_paras)
    # # print(lines_sorted_per_para)
    # reading_order_final[file_name]['lines_sorted_per_para'] = lines_sorted_per_para

    # # order = 0
    # # for i,para in enumerate(lines_sorted_per_para):
    # #     for line in para:
    # #         cv2.rectangle(img, (line[0],line[1]), (line[2],line[3]),2)
    # #         cv2.putText(img, str(order),(line[0],line[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
    # #         order+=1

    # words_sorted_per_line = get_word_order(word_json, lines_sorted_per_para)
    # # print(words_sorted_per_line)
    # reading_order_final[file_name]['words_sorted_per_line'] = words_sorted_per_line

    # # all_reading_orders[file_name] = reading_order_final[file_name]

    # # order = 0
    # # for i,para in enumerate(words_sorted_per_line):
    # #     for line in para:
    # #         cv2.rectangle(img, (line[0],line[1]), (line[2],line[3]),2)
    # #         cv2.putText(img, str(order),(line[0],line[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
    # #         order+=1

    # Create the directory if it doesn't exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Get the input image file name (without extension) from the first element in args.input
    file_name = os.path.splitext(os.path.basename(args.input[0]))[0]

    # Construct the full path for saving the image
    output_path = f'{output_folder_images}/{file_name}.jpg'  # Assuming you want to save as .png

    # Save the image
    cv2.imwrite(output_path, img)
    
    output_file = f'{output_folder}/{file_name}.json'
    with open(output_file, 'w') as f:
        json.dump(reading_order_final, f, indent=4)


Reading_Order_Generator(args)
