#!/usr/bin/env python
# coding: utf-8

# In[7]:


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np

'''
convert voc .xml to darknet format
it will create three files, train.txt, test.txt and labels
'''
classes = ["Car", "Bus", "Truck", "Pedestrian"]
image_folder='images/'
ann='annotations/'

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(filename):
    in_file = open(ann+filename+'.xml')
    out_file = open('labels/'+filename+'.txt', 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()
    
wd = getcwd()

if not os.path.exists('labels/'):
    os.makedirs('labels/')

train=open('train.txt','w')
test=open('test.txt','w')


for annos in os.listdir(ann):
    filename=annos[:-4]
    print(filename)
    convert_annotation(filename)
    isTrain = np.random.choice([0, 1], 1, p=[0.2, 0.8])[0]
    if isTrain:
        train.write(image_folder+filename+'\n')
    else:
        test.write(image_folder+filename+'\n')

train.close()
test.close()


# In[ ]:




