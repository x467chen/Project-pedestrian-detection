## Synopsis

In this project, I will try to  implement HOG+SVM and Faster R-CNN to do the simulations of pedestrain detection in  state-of-the-art system separately. And do some analysis and comparison with the experimental results based on the former techniques which have been introduced.


## Motivation

Why detection in self driving? Nowadays, with the explosive grows in population density and less limitation for people to own cars over the past decade has led to extensive state-of-the-art system research in recognition and detection traffic situation to promote a safer environment. By the leading of google, the research on the self-driving is growing vigorously, which also means the demands of the accuracy of pedestrian detection having been more and more urgent to make sure the safety situations during the self-driving. Also, the widely using property of the pedestrian detection (can not only be used in the self-driving system but also be widely used in some relative fields such as IDAS, intelligent surveillance and so on) makes scientists and companies keeping paying attention to the state-of-art system of the pedestrian detection. Furthermore, the governments also appear strong support for the development of the pedestrian detection in the recent years, because pedestrian detection can have a crucial contribution to the reduction of the collision rate caused by the human factors.

## Installation

A.HOG+SVM
1. Install Xcode Version 8.3.3<br>
2. Implement the OpenCV lib in Version 2.4.13.2. (Version 3.* is not recommanded)


B.Faster R-CNN
1. Install Anaconda-Navigator1.5 and configure your enviroment with: Keras2.0.6, Numpy, opencv-python2.4.11, sklearn0.18.2..<br>
2. Use the Pycharm Professional 2017 with the environment you configured in Anaconda-Navigator <br>
3. Llanguage of code is Python 2.7.0. <br>

C. Download Dataset
1. the links of Dataset  could be found inthe Contributors part.


## Code

1.The trained classifier is under HOG_SVM_Result/HOG-SVM-1.xml, HOG_SVM_Result/HOG-SVM-2.xml and HOG_SVM_Result/HOG-SVM-3.xml <br>
2.The most recently trained weight is stored I under the file Faster_R-CNN_Result/config.pickle and Faster_R-CNN_Result/model_frcnn.hdf5  <br>
3.Notice: Please change all the full path before running the code.


## Contributors

1.The Faster R-CNN Framework I used  can be found in the following link: https://github.com/rbgirshick/py-faster-rcnn<br>
2. The Data I used for train HOG+SVM is from INRIA Benchmark Data Set, which can be found in the following link: http://pascal.inrialpes.fr/data/human/ <br>
3.The Data I used for train Faster R-CNN is from the Visual Object Classes Challenge2012 (VOC2012), which can be found in the following link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ <br>
