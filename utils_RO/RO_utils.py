import pandas as pd
import numpy as np
import cv2
import os


def get_paragraph_order(component, image_file_name, output_path):
    '''
    Sorts the paragraphs based on the top-down and left-right order
    '''
    tlbr = component
    # tlbr = []
    # for idx, row in component.iterrows():
    #     tlbr.append([row['Left'][0], row['Top'][1], row['Right'][0], row['Bottom'][1]])
    tlbr_sorted_x = sorted(tlbr, key=lambda x: x[0])
    
    # mean_x = sum([box[0] for box in tlbr_sorted_x]) / len(tlbr_sorted_x)
    # median_x = tlbr_sorted_x[len(tlbr_sorted_x) // 2][0]
    
    mean_width = sum([box[2] - box[0] for box in tlbr_sorted_x]) / len(tlbr_sorted_x)
    median_width = tlbr_sorted_x[len(tlbr_sorted_x) // 2][2] - tlbr_sorted_x[len(tlbr_sorted_x) // 2][0]

    print("Mean width:", mean_width, "Median width:", median_width)
    # mean_x = min(mean_width, median_width)
    mean_x = mean_width
    # mean_x = median_width

    current_vert_line = tlbr_sorted_x[0][0]
    vert_lines = []
    temp_line = []

    for box in tlbr_sorted_x:
        if box[0] >= current_vert_line + (mean_x):
            vert_lines.append(temp_line)
            temp_line = [box]
            current_vert_line = box[0]
            continue
        temp_line.append(box)
    vert_lines.append(temp_line)
    print("Vertical lines")
    print(vert_lines)
    print(len(vert_lines))
    for line in vert_lines:
        line.sort(key=lambda x: x[1])
    
    return vert_lines
    # #visualise the paragraph order
    # image = cv2.imread(image_file_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    #set the order of the paragraphs in component and visualise
    # order=0
    # for line in vert_lines:
    #     for box in line:
    #         for idx, row in component.iterrows():
    #             if math.ceil(row['Left'][0])== math.ceil(box[0]) and math.ceil(row['Top'][1])== math.ceil(box[1]):
    #                 component.loc[idx, 'Order'] = order
    #                 cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    #                 cv2.putText(image, str(order), (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #                 order+=1

    #                     # Crop the image
    #                 cropped_image = image[box[1]:box[3], box[0]:box[2]]
    #                 # Save the cropped image
    #                 # uncomment the below line to save paragraph level images
    #                 # cv2.imwrite(f"cropped_{order}.jpg", cropped_image)
                

    # os.makedirs('{}'.format(output_path), exist_ok=True)                
    # output_path = '{}/{}_para_order.png'.format(output_path,os.path.basename(image_file_name).split('.')[0])
    # # output_path = '/home2/sreevatsa/paragraph_order.png'.format(os.path.basename(image_file_name).split('.')[0])
    # cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # #TODO: visualise_para_order() also visualises the paragraph order, so we modify and remove the above code as required

    # # print("Component after para order")
    # # print(component)  
    

    # return component

def calculate_overlap_percentage(large_box, small_box):

    large_x1, large_y1, large_x2, large_y2 = large_box
    small_x1, small_y1, small_x2, small_y2 = small_box

    overlap_x1 = max(large_x1, small_x1)
    overlap_y1 = max(large_y1, small_y1)
    overlap_x2 = min(large_x2, small_x2)
    overlap_y2 = min(large_y2, small_y2)

    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    large_area = (large_x2 - large_x1) * (large_y2 - large_y1)
    small_area = (small_x2 - small_x1) * (small_y2 - small_y1)

    # overlap_percentage = (overlap_area / large_area) * 100
    overlap_percentage = (overlap_area/small_area)*100
    return overlap_percentage

def is_box_inside(large_box, small_box, threshold_percentage):

    large_x1, large_y1, large_x2, large_y2 = large_box
    small_x1, small_y1, small_x2, small_y2 = small_box

    overlap_percentage = calculate_overlap_percentage(large_box, small_box)
    # print("Overlap:",overlap_percentage)
    if (large_x1 < small_x1 and small_x2 < large_x2 and large_y1 < small_y1 and small_y2 < large_y2):
        return True
    elif (overlap_percentage >= threshold_percentage):
        return True
    else:
        return False

def filter_paras_layout(para_bboxes, layout_bboxes):
    # regions_to_filter = ['figure', 'table', 'caption','figure-caption', 'formula', 'advertisement', 'page-number', 'page number' 'header', 'footer', 'supplementary', 'supplementary_note', 'headline','folio','sidebar','dateline','author','footnote','contact-info','chapter-title']
    regions_to_filter = ['figure', 'table', 'caption','figure-caption', 'formula', 'advertisement', 'page-number', 'page number' 'header', 'footer', 'supplementary', 'supplementary_note', 'headline','folio','sidebar','footnote','contact-info','chapter-title']
    filtered_paras = []
    
    for i, row in enumerate(para_bboxes['paras']):
        tlbr = [row[0], row[1], row[2], row[3]]
        keep_paragraph = True  # Flag to determine if the paragraph should be kept
        
        for key, values in layout_bboxes['layout'].items():
            if key in regions_to_filter:
                
                for value in values:
                    if is_box_inside(value, tlbr, threshold_percentage=75):
                        keep_paragraph = False
                        break  # Exit the loop if a match is found
                
                if not keep_paragraph:
                    break  # Exit the outer loop as well if a match is found
        
        if keep_paragraph:
            filtered_paras.append(row)  # Add the paragraph to the filtered list if it passes the check
    
    # Update para_bboxes with filtered paragraphs
    para_bboxes['paras'] = filtered_paras
    return para_bboxes

def get_line_order(line_bboxes, sorted_paras):
    # Flatten the sorted_paras list of lists to get all paragraph bounding boxes
    paras_boxes = [para for column in sorted_paras for para in column]
    
    paras_for_lines = []

    # Iterate over each paragraph box
    for para in paras_boxes:
        lines_in_para = []
        
        # Check each line to see if it is inside the current paragraph's bounding box
        for line in line_bboxes['lines']:
            if is_box_inside(para, line, threshold_percentage=100):  # Assuming 100% threshold means the line must be fully inside the para box
                lines_in_para.append(line)

        lines_in_para.sort(key=lambda line: line[1])
        paras_for_lines.append(lines_in_para)
    
    return paras_for_lines

def get_word_order(line_bboxes, sorted_paras):
    # Flatten the sorted_paras list of lists to get all paragraph bounding boxes
    paras_boxes = [para for column in sorted_paras for para in column]
    
    paras_for_lines = []

    # Iterate over each paragraph box
    for para in paras_boxes:
        lines_in_para = []
        
        # Check each line to see if it is inside the current paragraph's bounding box
        for line in line_bboxes['words']:
            if is_box_inside(para, line, threshold_percentage=90):  # Assuming 100% threshold means the line must be fully inside the para box
                lines_in_para.append(line)

        lines_in_para.sort(key=lambda line: line[0])
        paras_for_lines.append(lines_in_para)
    
    return paras_for_lines

def filter_boxes(boxes, threshold_percentage=90, type='words'):
    """
    Filters out words whose bounding boxes are significantly inside another word's bounding box.
    
    :param words: List of lists where each inner list consists of [x1, y1, x2, y2] for a word.
    :param threshold_percentage: Overlap percentage threshold to consider a word inside another.
    :return: Filtered list of bounding boxes.
    """
    filtered = []
    
    for i, small_box in enumerate(boxes):
        inside = False
        for j, large_box in enumerate(boxes):
            if i != j:  # Avoid comparing a box with itself
                if is_box_inside(large_box, small_box, threshold_percentage):
                    inside = True
                    break
        if not inside:
            filtered.append(small_box)
    
    # return {'words':filtered}
    return {f'{type}':filtered}

