import os
import sys
from xml.etree import ElementTree
from itertools import product
import urllib.request as request

import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import colorsys
import matplotlib.pyplot as plt

# ReNom version >= 2.3.0
from renom.cuda import set_cuda_active
from renom.algorithm.image.detection.yolo import build_truth, Yolo
from renom.utility.distributor import ImageDetectionDistributor
from renom.utility.image import *

set_cuda_active(True)

dataset_path = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/Annotations/"
novel_dataset_path = "/few_shot/dataset/novel/test_datah/"
batch_path = "/dataset/VOCtrainval12/batch.txt"
dataset_file = open(batch_path,'r')
train_file_list = [path.strip().split('.')[0] for path in dataset_file.readlines()]
#test_file_list = [path for path in os.listdir(dataset_path) if "2012_" in path]

#tree = ElementTree.parse(os.path.join(dataset_path, train_file_list[-1]))
#novel_classes_dirs = [x[0] for x in os.walk(novel_dataset_path)]

#Checking the dataset

def parse(node, indent=1):
    print("{}{} {}".format('    ' * indent, node.tag, node.text.strip()))
    for child in node:
        parse(child, indent + 1)
        
# print("/// Contents of a XML file ///")
# parse(tree.getroot())

#Getting the bounding box information

label_dict = {}
img_size = (416, 416)
cells = 13

def get_obj_coordinate(obj):
    global label_dict
    class_name = obj.find("name").text.strip()
    # if label_dict.get(class_name, None) is None:
    #     label_dict[class_name] = len(label_dict)
    label_dict = {'chair': 0, 'tvmonitor': 1, 'sofa': 2, 'bird': 3, 'person': 4, 'cat': 5, 'dog': 6, 'sheep': 7, 'car': 8, 'motorbike': 9,
    'bottle': 10, 'bus': 11, 'horse': 12, 'train' : 13, 'cow':14, 'diningtable' : 15 ,'boat': 16,'aeroplane':17,'bicycle':18,'pottedplant':19,
    'adidas1': 20, 'bershka': 21, 'calvinklein':22, 'chanel':23,'gucci':24,'lacoste':25}
    # 'audi': 21, 'BMW': 23,'chevrolet':26,'danone':27,'generalelectric':28,'samsung':31
    class_id = label_dict[class_name]
    bbox = obj.find("bndbox")
    xmax = float(bbox.find("xmax").text.strip())
    xmin = float(bbox.find("xmin").text.strip())
    ymax = float(bbox.find("ymax").text.strip())
    ymin = float(bbox.find("ymin").text.strip())
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w/2
    y = ymin + h/2
    return class_id, x, y, w, h
    
def get_img_info(filename):
    tree = ElementTree.parse(filename)
    node = tree.getroot()
    file_name = node.find("filename").text.strip()
    img_h = float(node.find("size").find("height").text.strip())
    img_w = float(node.find("size").find("width").text.strip())
    obj_list = node.findall("object")
    objects = []
    for obj in obj_list:
        objects.append(get_obj_coordinate(obj))
    return file_name, img_w, img_h, objects
        

train_data_set = []
#test_data_set = []

for o in train_file_list:
    if os.path.exists(os.path.join(dataset_path, o.split('/')[-1]+'.xml')):
        train_data_set.append(get_img_info(os.path.join(dataset_path, o.split('/')[-1]+'.xml')))
    else:
        train_data_set.append(get_img_info(os.path.join(o+'.xml')))
    

    
# for o in test_file_list:
#     test_data_set.append(get_img_info(os.path.join(dataset_path, o')))


#Making target Data

label_length = len(label_dict)
last_layer_size = cells*cells*(5*2+label_length)

def one_hot(label):
    oh = [0]*label_length
    oh[label] = 1
    return oh

def create_detection_distributor(train_set=True):
    img_path_list = []
    label_list = []
    if train_set:
        file_list = train_file_list
        data_set = train_data_set
        
    else:
        pass
        #file_list = test_file_list
        #data_set = test_data_set
  
    for i in range(len(file_list)):
        #img_path = os.path.join("/home/neox/Desktop/few_shot_learning/dataset/VOCtrainval12/VOCdevkit/VOC2012/JPEGImages/", data_set[i][0])
        img_path = file_list[i]+".jpg"
        # obj[1]:X, obj[2]:Y, obj[3]:Width, obj[4]:Height, obj[0]:Class
        objects = []
        for obj in data_set[i][3]:
            detect_label = {"bndbox":[obj[1], obj[2], obj[3], obj[4]],
                            "name":one_hot(obj[0])}
            objects.append(detect_label)
        img_path_list.append(img_path)
        label_list.append(objects)
    class_list = [c for c, v in sorted(label_dict.items(), key=lambda x:x[1])]
    return ImageDetectionDistributor(img_path_list,
                                     label_list,
                                     class_list,
                                     imsize = img_size)

def transform_to_yolo_format(label):
    yolo_format = []
    for l in label:
        yolo_format.append(build_truth(l.reshape(1, -1), img_size[0], img_size[1], cells, label_length).flatten())
    return np.array(yolo_format)

def draw_rect(draw_obj, rect):
    cor = (rect[0][0], rect[0][1], rect[1][0], rect[1][1])
    line_width = 3
    for i in range(line_width):
        draw_obj.rectangle(cor, outline="red")  
        cor = (cor[0]+1,cor[1]+1, cor[2]+1,cor[3]+1) 

train_detect_dist = create_detection_distributor(True)
#test_detect_dist = create_detection_distributor(False)

#Check Label Data

"""
sample, sample_label = train_detect_dist.batch(3, shuffle=True).__next__()
sampla, sampla_labela = train_detect_dist.batch(1, shuffle=True).__next__()
print("Hahahaha-Hihihih",sampla)
print("Bbox",sampla_labela[0][0:4])
print("label",np.argmax(sample_label[0][4:4+label_length]))
print("Hahahaha",label_dict)
for Mth_img in range(len(sample)):
    example_img = Image.fromarray(((sample[Mth_img]+1)*255/2).transpose(1, 2, 0).astype(np.uint8))
    dr = ImageDraw.Draw(example_img)

    print("///Objects")
    for i in range(0, len(sample_label[Mth_img]), 4+label_length):
        class_label = np.argmax(sample_label[Mth_img][i+4:i+4+label_length])
        x, y, w, h = sample_label[Mth_img][i:i+4]
        if x==y==h==w==0:
            break
        draw_rect(dr, ((x-w/2, y-h/2), (x+w/2, y+h/2)))
        print("obj:%d"%(i+1),
              "class:{:7s}".format([k for k, v in label_dict.items() if v==class_label][0]), 
              "x:%3d, y:%3d width:%3d height:%3d"%(x, y, w, h))

    # plt.figure(figsize=(4, 4))
    # plt.imshow(example_img)
    # plt.show()
for j, (img, label) in enumerate(train_detect_dist.batch(16, True)):
    yolo_format_label = transform_to_yolo_format(label)
    #print(yolo_format_label.shape)
    #print(yolo_format_label)
"""