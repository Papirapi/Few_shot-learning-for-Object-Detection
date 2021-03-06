from __future__ import division
import os
import numpy as np
import re
import cv2
import random
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.utils import to_categorical
from yolo_v2 import yolo_v2
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import subprocess
from postprocessing import yoloPostProcess
from yolov2_train import processGroundTruth
from yolov2_train import YoloLoss


#os.system("export CUDA_VISIBLE_DEVICES=''")


yolov2 = yolo_v2()
priors= np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]).reshape(5, 2)
config = tf.ConfigProto()
#device_count = {'GPU': 0 , 'GPU': 1 }
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)

label_dict = {'chair': 0, 'tvmonitor': 1, 'sofa': 2, 'bird': 3, 'person': 4, 'cat': 5, 'dog': 6, 'sheep': 7, 'car': 8, 'motorbike': 9,
    'bottle': 10, 'bus': 11, 'horse': 12, 'train' : 13, 'cow':14, 'diningtable' : 15 ,'boat': 16,'aeroplane':17,'bicycle':18,'pottedplant':19,
    'adidas1': 20, 'bershka': 21, 'calvinklein':22, 'chanel':23,'gucci':24,'lacoste':25}

base_dataset_rep = "/dataset/VOCtrainval12/"
base_data_dir = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/ImageSets/Main/"
novel_data_dir = "/dataset/novel/test_datah/"
base_dir_images = '/dataset/VOCtrainval12/VOCdevkit/VOC2012/JPEGImages/'
base_annotations_dir = "/dataset/VOCtrainval12/VOCdevkit/VOC2012/Annotations/"
files_classes = glob.glob(base_data_dir+"*_trainval.txt")
cfg_file_darknet = "/darknet/cfg/yolov2-voc.cfg"
cfg_file_darkflow = "/darkflow-master/cfg/yolov2-voc.cfg"
#==========================================LIST OF IMAGES IN EACH BASE CLASS============================================
k_shot = 10
all_images_list = []
classes_list = []
# Base classes
for f in files_classes:
    base_class = f.split('/')[-1].split('_')[0]
    classes_list.append(base_class)
    txt = open(f, 'r')
    images = []
    lines = txt.readlines()
    m = 0
    while len(images)<k_shot:
        line = lines[m]
        line = re.sub(r'\s{2,}', ' ', line.strip())
        l = line.split(" ")
        if l[1] == str(1):
            tree = ET.parse(base_annotations_dir + l[0] + '.xml')
            root = tree.getroot()
            objects = root.findall('object')
            names = np.array([obj.find('name').text for obj in objects])
            if (names == base_class).sum() == 1:
                images.append(base_dir_images + l[0]+'.jpg')
        m+=1
    all_images_list.append(images)
    #base_images_list.append(images)

# Novel classes
novel_classes = os.listdir(novel_data_dir)
classes_list.extend(novel_classes)
for novel in novel_classes:
    novel_images = glob.glob(novel_data_dir+novel+'/*.jpg')
    #novel_images_list.append(novel_images)
    all_images_list.append(novel_images)
all_images_train = [item for sublist in all_images_list for item in sublist]

# ===============================================hyperparameters=========================================================

batch_size = 16
lr = 0.0001
num_batches = round(len(all_images_train)/batch_size)
num_iterations = 15000
num_epochs = round(num_iterations/num_batches)
num_classes = 26
# ===================================================FUNCTIONS===========================================================
""" def base_novel_grinder(base_classes,novel_classes,base_images,novel_images):
    index_classes = random.sample(range(len(base_classes)),20)
    base_classes = (np.array(base_classes)[index_classes]).tolist()
    base_images = (np.array(base_images)[index_classes]).tolist()
    labels = dict((k,v) for k,v in label_dict.items() if v in index_classes or v in range(20,26))
    classes_list = base_classes+novel_classes
    images_list = base_images+novel_images
    return classes_list,images_list,labels """

def replace_line(file_name, text):
    lines = open(file_name, 'r').readlines()
    for n,l in enumerate(lines):
        if 'learning_rate' in l:
            line_num = n
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

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



def create_placeholders(n_H0, n_W0, n_C0, n_H1, n_W1, n_C1,n_cl,n_cell,n_box,n_res):

    X0 = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    X1 = tf.placeholder(tf.float32,[None, n_H1, n_W1, n_C1])
    Y = tf.placeholder(tf.float32,[None, n_cl,n_cell,n_cell,n_box,n_res])
    
    return X0, X1, Y

def forward_prop(meta_data,x_input_darknet,b_size):
    """
    input_shape: input shape of the image
    return: MetaModel
    """
    #conv2d
    Z1 = yolov2.conv_layer(meta_data, [3,3,4,32],padding='VALID')
    #relu
    A1 = tf.nn.relu(Z1)
    #max pool
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #conv2d
    Z2 = yolov2.conv_layer(P1, [3,3,32,64],padding='VALID')
    #relu
    A2 = tf.nn.relu(Z2)
    #max pool
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #conv2d
    Z3 = yolov2.conv_layer(P2, [30,30,64,1024],padding='VALID')
    #relu
    A3 = tf.nn.relu(Z3)
    reweights = tf.multiply(x_input_darknet,A3)
    net = yolov2.conv_layer(reweights, [3, 3, 1024, 1024])
    net = yolov2.conv_layer(net, [3, 3, 1024, 1024])
    net24 = yolov2.conv_layer(net, [3, 3, 1024, 1024])
    net = yolov2.conv_layer(net24, [3, 3, int(net24.get_shape()[3]), 1024])
    net = yolov2.conv_layer(net, [1, 1, 1024, 5 * (num_classes + 5)], batch_norm=False, name = 'out')
    predict = tf.reshape(net, [b_size,num_classes, 13, 13, 5, num_classes + 5])
    return predict

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
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

def meta_model(X_meta,X_yolo, Y_train, learning_rate=lr, num_epochs=1, minibatch_size=1, print_cost=True,first_iter=False):
    tf.reset_default_graph()
    (m,n_H0,n_W0,n_C0) = X_meta.shape
    (m,n_H1,n_W1,n_C1) = X_yolo.shape
    samp,n_cl,n_cell,n_cell,n_box,n_res = Y_train.shape
    num_minibatches = int(m/minibatch_size)
    costs = []
    
    X0,X1,Y = create_placeholders(n_H0,n_W0,n_C0,n_H1,n_W1,n_C1,n_cl,n_cell,n_box,n_res)
     
    Z4 = forward_prop(X0,X1,minibatch_size)
    print("Z4",Z4.shape)
    cost = YoloLoss(priors).loss(Y, Z4)
    #cost = yolov2.loss_layer(Z4, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9, beta2=0.999, epsilon=10 ** -9).minimize(cost)
     
    init = tf.global_variables_initializer()
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in all_variables:
        print(var)
    saver = tf.train.Saver()
    #Depend on your TF version you may not encounter beta1_power "biases_7/Adam_1 ....
    if first_iter:
        temp_saver = tf.train.Saver(var_list=[v for v in all_variables if all(elem not in v.name for elem in ["beta1_power_1","beta2_power_1", "biases_7/Adam_1","biases_7/Adam", "weight_7/Adam","weight_7/Adam_1", "weight_7", "biases_7"])])
    else:
        temp_saver = tf.train.Saver(var_list=[v for v in all_variables if all(elem not in v.name for elem in ["beta1_power_1","beta2_power_1"])])
    with tf.Session(config=config) as sess:
        #sess.run(init)
        sess.run(tf.variables_initializer(all_variables))
        temp_saver.restore(sess, "/darknet/models/param.ckpt")
        for epoch in range(num_epochs):
            epoch_cost = 0
            for l in range(num_minibatches):
                (minibatch_x, minibatch_y) = X_yolo[l].reshape(1,X_yolo.shape[1],X_yolo.shape[2],X_yolo.shape[3]),Y_train[l].reshape(1,Y_train.shape[1],
                Y_train.shape[2],Y_train.shape[3],Y_train.shape[4],Y_train.shape[5])
                #print("minibatches sizes", minibatch_x.shape,minibatch_y.shape)
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
        tf.train.write_graph(frozen_graph, "/darknet/models/", "my_model_ft.pb", as_text=False)
        return cost

    

def base_mask_channel(image, cls):
    """
    creating mask of image corresponding to class cls
    """
    im = cv2.imread(image)
    im = cv2.resize(im, (124, 124), interpolation=cv2.INTER_CUBIC)
    height, width, depth = im.shape
    tree = ET.parse(base_annotations_dir + (image.split('/')[-1]).split('.')[0] + '.xml')
    root = tree.getroot()
    objects = root.findall('object')
    names = [obj.find('name').text for obj in objects]
    ind = names.index(cls)
    obj = objects[ind]
    mask = np.zeros((height, width, depth + 1))
    bndbox = obj.findall('bndbox')[-1]
    xmin = int(bndbox[0].text)
    ymin = int(bndbox[1].text)
    xmax = int(bndbox[2].text)
    ymax = int(bndbox[3].text)
    mask[ymin:ymax + 1, xmin:xmax + 1, depth] = 1
    mask[:, :, :depth] = im

    return mask

def novel_mask_channel(image):
    im = cv2.imread(image)
    im = cv2.resize(im, (124, 124), interpolation=cv2.INTER_CUBIC)
    height, width, depth = im.shape
    mask = np.zeros((height, width, depth+1))
    tree = ET.parse(img[:-4]+'.xml')
    root = tree.getroot()

    object = root.find('object')
    bndbox = object.find('bndbox')
    xmin = int(bndbox[0].text)
    ymin = int(bndbox[1].text)
    xmax = int(bndbox[2].text)
    ymax = int(bndbox[3].text)
    mask[ymin:ymax+1, xmin:xmax+1, depth] = 1
    mask[:, :, :depth] = im
    return mask

# ========================================================TRAINING=======================================================
init = 0
pre_weights = "/darknet/backup/yolov2-voc.weights"
for epoch in range(num_epochs):
    random.shuffle(all_images_train)
    for i in range(num_batches):
               
        print("===============================iteration"+str(i)+"========================================")
        batch_train = open(base_dataset_rep+"batch.txt", 'w')
        for j in range(init, init+batch_size):
            batch_train.write(str(all_images_train[j])+'\n')
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
        batch_train = open(base_dataset_rep + "batch.txt", 'r' ,encoding='utf-8-sig')

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
        from dataprepa_few import *
        
        #label_length = len(rand_labels)
        train_detect_dist = create_detection_distributor(True)
        (sample, sample_label) = train_detect_dist.batch(batch_size, shuffle=True).__next__() 
        y_true = np.zeros((batch_size,num_classes,13, 13, 5, num_classes+5))
        
        for Mth_img in range(len(sample)):
            y_sample = np.zeros((num_classes,13,13,5,num_classes+5))
            for l in range(num_classes):
                bounding_boxes , labels = [],[]
                for k in range(0, len(sample_label[Mth_img]), 4+label_length):
                    label = np.argmax(sample_label[Mth_img][k+4:k+4+label_length])
                    x, y, w, h = sample_label[Mth_img][k:k+4]
                    if x==y==h==w==0.0:
                        break
                    if label == l:
                        labels.append(one_hot(np.argmax(sample_label[Mth_img][k+4:k+4+label_length])))
                        bounding_boxes.append([x, y, w, h])
                if labels!=[]:       
                    y_class = processGroundTruth(np.array(bounding_boxes), labels, priors, (13, 13, 5, num_classes+5))
                    y_sample[l] = y_class
            y_true[Mth_img] = y_sample

        # ************************************* META MODEL ********************************************************
        #Preparing the train data 
        meta_train_x = []
        meta_train_y = np.arange(num_classes)
        meta_train_y = to_categorical(meta_train_y,num_classes)
        #classes_list,images_list,rand_labels = base_novel_grinder(base_classes_list,novel_classes_list,base_images_list,novel_images_list)
        for n, cl in enumerate(classes_list):
            img = random.choice(all_images_list[n])
            if cl in novel_classes:
                mask = novel_mask_channel(img)
            else:
                mask = base_mask_channel(img, cl)
            meta_train_x.append(mask)
        meta_train_x = np.array(meta_train_x)
        #Training the metaModel
        print('y_true.shape',y_true.shape)
        meta_train_tensor = tf.convert_to_tensor(meta_train_x, dtype=tf.float32)

        if i==0:
            meta_model(meta_train_x,y_out, y_true, learning_rate=lr, num_epochs=1, minibatch_size=1, print_cost=True,first_iter=True)
        else:
            meta_model(meta_train_x,y_out, y_true, learning_rate=lr, num_epochs=1, minibatch_size=1, print_cost=True,first_iter=False)
    
