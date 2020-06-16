import os
import cv2

vids=os.listdir('vids')

for vid in vids:
    x=vid

    vidObj=cv2.VideoCapture('vids/'+x)


    success=1
    count=0

    os.mkdir(x[:-4])

    while success:
        success,img=vidObj.read()
        cv2.imwrite(x[:-4]+'/frame__%d.jpg' % count, img)
        count+=1