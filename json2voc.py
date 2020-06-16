#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
from xml.etree.ElementTree import Element,SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os
import shutil

'''
convert our json data to OVC xml format
segmented, truncated, difficult, pos are not shown in our data, so I set then to 0 as default
'''

segmented=0
truncated = 0
difficult = 0
pose = 'Unspecified'

def set_annotation(w,h,f,fn):
    top = Element('annotation')
    top.set('verified', 'Yes')
    folder = SubElement(top, 'folder')
    folder.text = f
    filename=SubElement(top,'filename')
    filename.text=fn
    source = SubElement(top, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    size = SubElement(top, 'size')
    width = SubElement(size, 'width')
    width.text = str(w)
    height = SubElement(size, 'height')
    height.text = str(h)
    depth = SubElement(size, 'depth')
    depth.text = str(3)
    seg = SubElement(top, 'segmented')
    seg.text = str(0)
    return top

def set_object(top,name_,xmin_,ymin_,xmax_,ymax_):
    object=SubElement(top,'object')
    name=SubElement(object,'name')
    name.text=name_
    pose=SubElement(object,'pose')
    pose.text='Unspecified'
    truncated=SubElement(object,'truncated')
    truncated.text=str(0)
    difficult=SubElement(object,'difficult')
    difficult.text=str(0)
    bndbox=SubElement(object,'bndbox')
    xmin=SubElement(bndbox,'xmin')
    xmin.text=str(xmin_)
    ymin=SubElement(bndbox,'ymin')
    ymin.text=str(ymin_)
    xmax=SubElement(bndbox,'xmax')
    xmax.text=str(xmax_)
    ymax=SubElement(bndbox,'ymax')
    ymax.text=str(ymax_)
    return top

def save_xml(content,filename):
    f=open(filename,'w+')
    f.write(content)
    f.close()



#convert json to xml
f='images' #the folder we store renamed images
root='YOUR ROOT FOLDER' #the folder of original data
four_data=['0','1','2','3'] #I rename very long names into shorter names
#ann_img=['/ann/','/img/'] #not used
annotation_folder='annotations/' #the folder we store new annotations
if not os.path.exists('labels/'):
    os.mkdir(annotation_folder)

#convert json to voc xml format
ann=os.path.join(root,'annotations/')
for annotation in os.listdir(ann):
    json_file_path=os.path.join(ann,annotation)
    xml_file_name=annotation_folder+annotation[:-5]+'.xml'
    fn=annotation[:-5]
    
    with open(json_file_path) as json_file:
        data=json.load(json_file)
        width,height,depth=data['size']['width'],data['size']['height'],3
        objects=data['objects']
        top = set_annotation(width, height, f, fn)
        for obj in objects:
            name = obj['classTitle']
            xmin = obj['points']['exterior'][0][0]
            ymin = obj['points']['exterior'][0][1]
            xmax = obj['points']['exterior'][1][0]
            ymax = obj['points']['exterior'][1][1]
            top=set_object(top,name,xmin,ymin,xmax,ymax)

        rough_string=ElementTree.tostring(top, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        res=reparsed.toprettyxml(indent="    ")
        save_xml(res[23:], xml_file_name)

#rename images in original folder and copy to new folder
image_folder='images/'
img = os.path.join(root, 'images/')
for image in os.listdir(img):
    image_file_path=os.path.join(img,image)
    new_image_file_path=image_folder+image
    shutil.copy(image_file_path,new_image_file_path)


# In[ ]:




