""" 
This script is used to draw paras from word detections on the images.
The code is for paragraphs but the names of the variables or functions are reused from lines
(ik bad code but it works)
"""

import random
import os
import cv2
import json
import numpy as np

line_colour = (0, 0, 255)
word_colour = (144, 238, 144)

# folder_path = "/ssd_scratch/sreevatsa/Reading-Order-Dataset/final_images_combined_extended"
# folder_path = "/ssd_scratch/sreevatsa/Reading-Order-Dataset/extended_dataset"
folder_path = "/ssd_scratch/sreevatsa/Reading-Order-Dataset/challenging"
# jsons_path = "/ssd_scratch/sreevatsa/Hi-SAM/RO_877/jsonl"
# jsons_path = "/ssd_scratch/sreevatsa/Hi-SAM/extended_RO_dataset/jsonl"
jsons_path = "/ssd_scratch/sreevatsa/Hi-SAM/Chinese/.jsonl"
results_path = "/ssd_scratch/sreevatsa/sreevatsa_hisam_lines_paras/results/lines"

if not os.path.exists(results_path):
    os.makedirs(results_path)

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

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
        words_in_line = []
        for line in paragraph['lines']:
            for word in line['words']:
                list_coordinate = word['vertices']
                tuple_coordinate = (list_coordinate[0], list_coordinate[1], list_coordinate[2], list_coordinate[3])
                words_in_line.append(tuple_coordinate)
        boxes_per_line.append(words_in_line)
    return boxes_per_line

def extract_polygons_from_json_per_line(json_path):
    # with open(json_path, 'r') as f:
    #     data = json.load(f)
    data = json_path
    polygons_per_line = []
    for paragraph in data:
        polygons_in_line = []
        for line in paragraph['lines']:
            for word in line['words']:
                polygon_list = word['vertices']
                polygons_in_line.append(polygon_list)
        polygons_per_line.append(polygons_in_line)
    return polygons_per_line


def multiple_folder_evaluate():

    for folder in os.listdir(folder_path):
        folder_name = folder
        folder = os.path.join(folder_path, folder)
        json_folder = os.path.join(jsons_path, folder_name, ".jsonl")
        results_folder = os.path.join(results_path, folder_name)
        os.makedirs(results_folder, exist_ok=True)
        
        for file in os.listdir(folder):
            file_name = file.split(".")[0]
            image_path = os.path.join(folder, file)
            json_path = os.path.join(json_folder, file_name + ".jsonl")
            
            image = cv2.imread(image_path)
            if check_bbox_or_polygon(json_path) == "bbox":
                boxes_per_line = extract_boxes_from_json_per_line(json_path)
                for boxes in boxes_per_line:
                    mask = image.copy()
                    for box in boxes:
                        # display the boxes as masks
                        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), word_colour, -1)
                    alpha = 0.4
                    image = cv2.addWeighted(image, alpha, mask, 1-alpha, 0)
                    
                    min_x = min([box[0] for box in boxes])
                    min_y = min([box[1] for box in boxes])
                    max_x = max([box[2] for box in boxes])
                    max_y = max([box[3] for box in boxes])
                    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), line_colour, 2)

            else:
                polygons = extract_polygons_from_json_per_line(json_path)
                for polygons_in_line in polygons:
                    colour = random_color()
                    for polygon in polygons_in_line:
                        points = np.array(polygon, dtype=np.int32)
                        points = points.reshape((-1, 1, 2))
                        cv2.polylines(image, [points], isClosed=True, color=colour, thickness=2)
            
            cv2.imwrite(os.path.join(results_folder, file), image)
            print(f"Processed {file}")
            
    print("Done")

import json
import os
import cv2

def single_folder_evaluate_paras(args,paragraphs):
    dict = {}
    dict['paras'] = {}
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
    dict['paras'] = bboxes
        
    
    # # with open('/ssd_scratch/sreevatsa/sreevatsa_hisam_lines_paras/final_jsons/lines_paragraphs_extended_RO_dataset.json','w') as json_file:
    # with open('/ssd_scratch/sreevatsa/sreevatsa_hisam_lines_paras/final_jsons/lines_paragraphs_challenging_RO_dataset.json','w') as json_file:    
    #     json.dump(dict,json_file)
    
    return dict



# call the appropriate function based on requirement

# single_folder_evaluate()
# multiple_folder_evaluate()


