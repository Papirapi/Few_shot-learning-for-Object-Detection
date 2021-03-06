# -*- coding:utf-8 -*-

from __future__ import division

import cv2
import glob
import numpy as np
import os
import re
import random
import subprocess
import tensorflow as tf
import xml.etree.ElementTree as ET

from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from postprocessing import yoloPostProcess
from tensorflow.python.framework import ops
from yolo_v2 import yolo_v2
from yolov2_train import processGroundTruth, YoloLoss

#os.system("export CUDA_VISIBLE_DEVICES=''")

yolov2 = yolo_v2()
priors= np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]).reshape(5, 2)
config = tf.ConfigProto()
config.log_device_placement = False  # to log device placement (on which device the operation ran)

dataset_rep = "/dataset/VOCtrainval12/"
dir_src = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/ImageSets/Main/"
dir_images = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/JPEGImages/"
annotations_dir = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/Annotations/"
files_classes = glob.glob(dir_src+"*_trainval.txt")
cfg_file_darknet = "/darknet/cfg/yolov2-voc.cfg"
cfg_file_darkflow = "/darkflow-master/cfg/yolov2-voc.cfg"
#==========================================LIST OF IMAGES IN EACH BASE CLASS============================================
images_list = []
classes_list = []

for f in files_classes:
    base_class = f.split('/')[-1].split('_')[0]
    classes_list.append(base_class)
    txt = open(f, 'r')
    images = []
    #class_dict.setdefault(base_class, [])
    for line in txt.readlines():
        line = re.sub(r'\s{2,}', ' ', line.strip())
        l = line.split(" ")
        if l[1] == str(1):
            images.append(l[0]+ '.jpg')
    images_list.append(images)
# ===============================================hyperparameters=========================================================
train_file = open(dataset_rep+"voc_train.txt", 'r',encoding='utf-8-sig')
images_train = train_file.readlines()
batch_size = 20
lr = 0.0001
num_batches = round(len(images_train)/batch_size)
num_iterations = 8000
num_epochs = round(num_iterations/num_batches)
num_classes = len(classes_list)
# ===================================================FUNCTIONS===========================================================

def create_placeholders(n_H0, n_W0, n_C0, n_H1, n_W1, n_C1,n_cl,n_cell,n_box,n_res):

    X0 = tf.placeholder(tf.float32,[None, n_cl, n_H0, n_W0, n_C0])
    X1 = tf.placeholder(tf.float32,[None, n_H1, n_W1, n_C1])
    Y = tf.placeholder(tf.float32,[None, n_cl,n_cell,n_cell,n_box,n_res])
    
    return X0, X1, Y

def forward_prop(meta_data,x_input_darknet,b_size = batch_size):
    """
    input_shape: input shape of the image
    return: MetaModel
    """
    #conv2d
    Z1 = yolov2.conv_layer(meta_data, [1,3,3,4,32],padding='VALID')
    print("Z1",Z1.get_shape())
    #relu
    A1 = tf.nn.relu(Z1)
    print("A1",A1.get_shape())
    #max pool
    P1 = tf.nn.max_pool3d(A1, ksize=[1,1,2,2,1], strides=[1,1,2,2,1], padding='SAME')
    print("P1",P1.get_shape())
    #conv2d
    Z2 = yolov2.conv_layer(P1, [1,3,3,32,64],padding='VALID')
    print("Z2",Z2.get_shape())
    #relu
    A2 = tf.nn.relu(Z2)
    print("A2",A2.get_shape())
    #max pool
    P2 = tf.nn.max_pool3d(A2, ksize=[1,1,2,2,1], strides=[1,1,2,2,1], padding='SAME')
    print("P2",P2.get_shape())
    #conv2d
    Z3 = yolov2.conv_layer(P2, [1,30,30,64,1024],padding='VALID')
    print("Z3",Z3.get_shape())
    #relu
    A3 = tf.nn.relu(Z3)
    print("A3",A3.get_shape())
    reweights = tf.multiply(x_input_darknet,A3)
    net = yolov2.conv_layer(reweights, [1,3, 3, 1024, 1024])
    net = yolov2.conv_layer(net, [1,3, 3, 1024, 1024])
    net24 = yolov2.conv_layer(net, [1,3, 3, 1024, 1024])
    net = yolov2.conv_layer(net24, [1,3, 3, int(net24.get_shape()[4]), 1024])
    net = yolov2.conv_layer(net, [1,1, 1, 1024, 5 * (1 + 5)], batch_norm=False, name = 'out')
    print('**************ahaya net', net.get_shape())
    
    return net

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def load_graph(frozen_graph_filename):
    """
    We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    """
    #ops.reset_default_graph()
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def mask_channel(image, cls):
    """
    creating mask of image corresponding to class cls
    """
    im = cv2.imread(dir_images + image )
    im = cv2.resize(im, (124, 124), interpolation=cv2.INTER_CUBIC)
    height, width, depth = im.shape
    tree = ET.parse(annotations_dir + image.split('.')[0] + '.xml')
    root = tree.getroot()
    masks_list = []
    objects = root.findall('object')
    for obj in objects:
        name = obj.find('name').text
        if name == cls:
            mask = np.zeros((height, width, depth + 1))
            bndbox = obj.findall('bndbox')[-1]
            xmin = int(bndbox[0].text)
            ymin = int(bndbox[1].text)
            xmax = int(bndbox[2].text)
            ymax = int(bndbox[3].text)
            mask[ymin:ymax + 1, xmin:xmax + 1, depth] = 1
            mask[:, :, :depth] = im
            masks_list.append(mask)

        else:
            pass
    return masks_list

def meta_model(X_meta,X_yolo, Y_train, learning_rate=lr, num_epochs=1, minibatch_size=1, print_cost=True,first_iter=False):
    tf.reset_default_graph()
    (m,n_cl,n_H0,n_W0,n_C0) = X_meta.shape
    (m,n_H1,n_W1,n_C1) = X_yolo.shape
    samp,n_cl,n_cell,n_cell,n_box,n_res = Y_train.shape
    num_minibatches = int(m/minibatch_size)
    costs = []
    X0,X1,Y = create_placeholders(n_H0,n_W0,n_C0,n_H1,n_W1,n_C1,n_cl,n_cell,n_box,n_res)
    
    Z4 = forward_prop(X0,X1,minibatch_size)
    #Z4 = tf.reshape(net, [minibatch_size,n_cl, 13, 13, 5, 1 + 5])
    print("Z4",Z4.shape)
    print('Y',Y.shape)
    cost = YoloLoss(priors).loss(Y, Z4)
    print('cost',cost.get_shape())
    #cost = yolov2.loss_layer(Z4, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9, beta2=0.999, epsilon=10 ** -9).minimize(cost)
    
    init = tf.global_variables_initializer()
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    temp_saver = tf.train.Saver(var_list=[v for v in all_variables if all(elem not in v.name for elem in ["beta1_power_1","beta2_power_1"])])
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        if first_iter:
            sess.run(init)
        else:
            sess.run(tf.variables_initializer(all_variables))
            temp_saver.restore(sess, "/darknet/models/param.ckpt")
        for epoch in range(num_epochs):
            epoch_cost = 0
            for l in range(num_minibatches):
                (minibatch_x, minibatch_y) = X_yolo[(l*minibatch_size):(l*minibatch_size)+minibatch_size].reshape(minibatch_size,X_yolo.shape[1],X_yolo.shape[2],X_yolo.shape[3]),Y_train[(l*minibatch_size):(l*minibatch_size)+minibatch_size].reshape(minibatch_size,Y_train.shape[1],
                Y_train.shape[2],Y_train.shape[3],Y_train.shape[4],Y_train.shape[5])
                print("minibatches sizes", minibatch_x.shape,minibatch_y.shape)
                _,minibatch_cost = sess.run([optimizer, cost], feed_dict={X0: X_meta, X1:minibatch_x, Y:minibatch_y})
                print("--------------------------------batch"+str(l)+"-------------------------------")
                print("minibatch_cost", minibatch_cost)
                epoch_cost= epoch_cost+(minibatch_cost/num_minibatches)
            if print_cost==True:
                print('cost after epoch {}:{}'.format(epoch, epoch_cost))
                costs.append(epoch_cost)
            
        saver.save(sess, "/darknet/models/param.ckpt")#save the parameters

        print('parameters have been saved')
        frozen_graph = freeze_session(sess, output_names=["out"])
        tf.train.write_graph(frozen_graph, "/darknet/models/", "my_model.pb", as_text=False)
        return cost

def replace_line(file_name, text):
    lines = open(file_name, 'r').readlines()
    for n,l in enumerate(lines):
        if 'learning_rate' in l:
            line_num = n
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

# ========================================================TRAINING=======================================================

pre_weights = "darknet53.conv.74"
for epoch in range(num_epochs):
    init = 0
    print("===============================Epoch"+str(epoch)+"========================================")
    for i in range(num_batches):
        # learning rate decay        
        iter = ((epoch-1)*num_batches)+i
        if iter == 500:
            replace_line(cfg_file_darknet, 'learning_rate=0.001\n')
            replace_line(cfg_file_darkflow, 'learning_rate=0.001\n')
            lr = 0.001
        elif iter == 4000:
            replace_line(cfg_file_darknet, 'learning_rate=0.0001\n')
            replace_line(cfg_file_darkflow, 'learning_rate=0.0001\n')
            lr = 0.0001
        elif iter == 6000:
            replace_line(cfg_file_darknet, 'learning_rate=0.00001\n')
            replace_line(cfg_file_darkflow, 'learning_rate=0.00001\n')
            lr = 0.00001
        else:
            pass
        
        print("===============================iteration"+str(i)+ "  Epoch"+str(epoch)+"========================================")
        batch_train = open("/home/ubuntu/Desktop/few_shot/dataset/VOCtrainval12/batch.txt", 'w')
        for j in range(init, init+batch_size):
            batch_train.write(str(images_train[j]))
        init += batch_size
        batch_train.close()
        # *************************************FEATURE EXTRACTOR (Yolo-v2)*******************************************************
        darknet_train = subprocess.Popen(["./darknet", "detector", "train", "cfg/voc.data", cfg_file_darknet, pre_weights])
        stdout, stderr = darknet_train.communicate()
        print(stdout)
        #os.rename("path ->  darknet/backup/yolov2-voc_final_"+"iter_"+str(i)+".weights")
        os.rename("/darknet/backup/yolov2-voc_final.weights",
                  "/darknet/backup/yolov2-voc.weights")
        darknet_weights = os.system(
            "../darkflow-master/flow --model "+cfg_file_darkflow+" --load /darknet/backup/yolov2-voc.weights --savepb")
        graph = load_graph("./built_graph/yolov2-voc.pb")
        

        x = graph.get_tensor_by_name('prefix/input:0')
        y = graph.get_tensor_by_name('prefix/39-convolutional:0')
        batch_train = open("/dataset/VOCtrainval12/batch.txt", 'r' ,encoding='utf-8-sig')

        all_images = []

        for image in batch_train.readlines():
            image = image.strip()
            img = load_img(image, target_size=(416, 416))
            img = img_to_array(img)
            img_matrix = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
            all_images.append(img_matrix)
        data = np.array(all_images)

        with tf.Session(graph=graph, config=config) as sess_graph:
            y_out = sess_graph.run(y, feed_dict={x: data})
        pre_weights = "/darknet/backup/yolov2-voc.weights"
        os.remove("./built_graph/yolov2-voc.pb")
        os.remove("./built_graph/yolov2-voc.meta")

        # *************************************Target Data Prep*******************************************************
        from dataprepa import *
        train_detect_dist = create_detection_distributor(True)
        (sample, sample_label) = train_detect_dist.batch(batch_size, shuffle=True).__next__()
        print('*****sample****',sample.shape)
        print('*****sample_label****',sample_label.shape)
        y_true = np.zeros((batch_size,num_classes,13, 13, 5, 5+1))
        
        for Mth_img in range(len(sample)):
            y_sample = np.zeros((num_classes,13,13,5,5+1))
            for l in range(num_classes):
                bounding_boxes , labels = [],[]
                for k in range(0, len(sample_label[Mth_img]), 4+label_length):
                    label = np.argmax(sample_label[Mth_img][k+4:k+4+label_length])
                    x, y, w, h = sample_label[Mth_img][k:k+4]
                    if x==y==h==w==0.0:
                        break
                    if label == l:
                        #labels.append(one_hot(label))
                        labels.append([1])
                        bounding_boxes.append([x, y, w, h])
                             
                if labels!=[]:       
                    y_class = processGroundTruth(np.array(bounding_boxes), labels, priors, (13, 13, 5, 5+1))
                    y_sample[l] = y_class
            y_true[Mth_img] = y_sample

        # ************************************* META MODEL ********************************************************
        #Preparing the train data 
        meta_train_x = []
        meta_train_y = np.arange(num_classes)
        meta_train_y = to_categorical(meta_train_y,num_classes)
        for n, cl in enumerate(classes_list):
            img = random.choice(images_list[n])
            masks_list = mask_channel(img, cl)
            mask = random.choice(masks_list)
            meta_train_x.append(mask)
        meta_train_x = np.array(meta_train_x)
        meta_train_x = meta_train_x.reshape(1,meta_train_x.shape[0],meta_train_x.shape[1],meta_train_x.shape[2],meta_train_x.shape[3])
        #Training the metaModel
        print('meta_train_x.shape',meta_train_x.shape)
        meta_train_tensor = tf.convert_to_tensor(meta_train_x, dtype=tf.float32)
        #metaModel = meta_model(meta_train_tensor,(y_out.shape[1],y_out.shape[2],y_out.shape[3]))

        if i==0:
            meta_model(meta_train_x,y_out, y_true, learning_rate=lr, num_epochs=1, print_cost=True,first_iter=True)
        else:
            meta_model(meta_train_x,y_out, y_true, learning_rate=lr, num_epochs=1, print_cost=True,first_iter=False)
    train_file.close()
    
