Those scripts may help if you use Darknet.

The json2voc.py is used to convert our json annotation file into VOC xml annotation.
The json file should look like this:

<img src=images/Capture.PNG>

All you need to do is to specify the path of images and annotations.

The doc2unix format.ipynb is used for formatting docs in Windows to unix in Linux, I create train.txt and test.txt in my local Windows
machine and when I upload them to GCP, the darknet cannot find training and testing data with these two .txt files, because there is a formatting 
issue, and Darknet will not tell you what happened, images are in the specified folder, but Darknet cannot read it. This code will wrok inplace

The change_zero.ipynb is used before you train the YOLO in Darknet. Darknet does not like zeros in the ground truth bounding box, it
is possible to have some objects at the top right corner thus the x,y coordinates are 0, if you train the YOLO with these data, there will be 
Nan everywhere, so this code will replace all 0s to 1s.

The darknet2maskrcnn.py is to transofrm the prediction result in AlexyAB-Darknet into our data format maskrcnn, this code dose not convert the result into standarn maskrcnn format, but with some simplification which is usable for evaluation.

The evaluation.py is a sample code to show you how to load data and get the performance of your model

The video2frame.py is to cut videos into frames.



