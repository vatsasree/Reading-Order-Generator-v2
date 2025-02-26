import cv2
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import json
import tqdm
import argparse

parser = argparse.ArgumentParser(description = 'Yolo inference')
parser.add_argument('--folder_path', type=str, default=None, help='Path to the folder containing images')
parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
parser.add_argument('--output_path', type=str, default=None, help='Path to the output folder')
parser.add_argument('--exp_name', type=str, default=None, help='Name of the experiment')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
save_path = os.path.join(args.output_path, args.exp_name)
os.makedirs(save_path, exist_ok=True)

# Load a pretrained YOLOv8n model
# model = YOLO("/home2/sreevatsa/py-scripts/yolov8_last.pt")
# model = YOLO('/home2/sreevatsa/py-scripts/yolo_1024_best.pt')
model = YOLO(args.model_path)
model = model.to('cuda:1')
dict={}

# folder_path = '/home2/sreevatsa/test_images/textbooks'
folder_path = args.folder_path
imglist = os.listdir(folder_path)
# imglist = 
imglist = [os.path.join(folder_path, i) for i in imglist]
# imglist = [os.path.join(folder_path, str(i)+'.jpg') for i in range(231,251)]

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
            dict[img_name.split('/')[-1].split('.')[0]] = temp_dict
    except Exception as e:
        print(f'Error in {img_name} : {e}')
        continue
    
    # print(dict)

print(dict)
for key,value in dict.items():
    # print(key)
    for k,v in value.items():
        for i in v:
            cv2.rectangle(source, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 1)
            cv2.putText(source, k, (int(i[0]), int(i[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    os.makedirs(os.path.join(save_path, 'imgs'), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, 'imgs', '{}_recon.jpg'.format(key)), source)

#save json
os.makedirs(os.path.join(save_path, 'json'), exist_ok=True)
with open(os.path.join(save_path, 'json', '{}.json'.format(args.exp_name)),'w') as f:
    json.dump(dict, f)


# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # # Show results to screen (in supported environments)
#     # r.show()

#     # Save results to disk
#     os.makedirs("/home2/sreevatsa/yoloresults", exist_ok=True)
#     r.save(filename=f"/home2/sreevatsa/yoloresults/{i}.jpg")