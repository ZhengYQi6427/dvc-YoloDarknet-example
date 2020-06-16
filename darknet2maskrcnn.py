import os
import json
import re
'''
below is the maskrcnn annotation format,use this code to convert darknet prediction to maskrcnn
{
1.0: {	# frame ID (numeric)
"class_ids": [1, 1, 2], # list of int, indexing from the list:
 [“background”, “pedestrian”, “vehicle”]
"scores": [0.99, 0.98, 0.87], # list of float
"contours": ...,		 # we don’t use this
"rois": [			 # list of list of numeric, as follows:
				 [bbox top left x coord, top left y,
 bottom right x, bottom right y]
[783, 562, 70, 149], ...
] 
}, ...
}
'''


'''
fn not the video, fn is the folder containing frames of this video, use video2frame.py to cut videos into frames, there should also be
a result.txt that contains prediction result of all frames using AlexyAB_Darknet, I have shown these steps in the Readme.md
'''
fn='traffic_video_GP020614_190720_0237_0407_90sec_calibrated'

d={}

'''
as you can see it's trying to open result.txt under the directory fn, and identifying Vehicle and Pedestrian in this file
'''
with open(fn+'/result.txt') as file:
    content=file.readlines()[4:-1]
    for line in content:
        if 'Enter Image Path' in line:
            filename=line.split(':')[1].split('/')[-1]
            frame_id=float(re.findall(r'\d+',filename)[0])
            d[frame_id]={'class_ids':[],
                         'scores':[],
                         'rois':[]}

        else:
            if 'Vehicle' in line:
                c, xtl, ytl, w, h = [float(x) for x in re.findall(r'\d+', line)]
                c=c/100
                xbr=xtl+w
                ybr=ytl+h
                d[frame_id]['class_ids'].append(2)
                d[frame_id]['scores'].append(c)
                d[frame_id]['rois'].append([xtl,ytl,xbr,ybr])
            elif 'Pedestrian' in line:
                c, xtl, ytl, w, h = [float(x) for x in re.findall(r'\d+', line)]
                c = c / 100
                xbr = xtl + w
                ybr = ytl + h
                d[frame_id]['class_ids'].append(1)
                d[frame_id]['scores'].append(c)
                d[frame_id]['rois'].append([xtl, ytl, xbr, ybr])
            else:
                print('WTF? '+line)

'''
the output will be a big .json file in maskrcnn format
'''
with open(fn+'/'+fn+'_pred.json','w') as fp:
    json.dump(d,fp)

