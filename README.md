# Remote-sensing-scene-classification
For the remote sensing scene classification task in complex background, we proposed a lightweight convolutional neural network with bilinear feature extraction structure. The idea of branch feature fusion was proposed to fuse the extracted features of different branches.


# Files
models--The folder that contains the model. The model.py is the proposed model code file.


original_data--Four remote sensing scene datasets without scale division.The download address of these four datasets is provided in this folder.


train.py--Train the model


classfication.py--Data analysis of training and test results


test.py--Test the classification results on the remote sensing scene data sets


split_data.py--Divide training set and test set


predict.py--The prediction of classification results


# Code execution environment
python 3.5.2


numpy 1.16.2


keras 2.1.3


tensorflow-gpu 1.4


opencv-python


matplotlib


pandas


seaborn


sklearn

We provide a virtual environment to support the project, which can be downloaded from the following link:

link:https://pan.baidu.com/s/1ZRum81EDRF6AbOMWkf8qhw 
password:g5ty

If this project is useful to you, please cite the following paper：

C. Shi, T. Wang and L. Wang, "Branch Feature Fusion Convolution Network for Remote Sensing Scene Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2020.3018307.


