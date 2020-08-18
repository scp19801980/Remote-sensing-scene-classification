# Remote-sensing-scene-classification
For the remote sensing scene classification task in complex background, we proposed a lightweight convolutional neural network with bilinear feature extraction structure. The idea of branch feature fusion was proposed to fuse the extracted features of different branches.\n
models--The folder that contains the model. The model.py is the proposed model code file./n
original_data--Four remote sensing scene datasets without scale division.The download address of these four datasets is provided in this folder./n
train.py--Train the model/n
classfication.py--Data analysis of training and test results/n
test.py--Test the classification results on the remote sensing scene data sets/n
split_data.py--Divide training set and test set/n
predict.py--The prediction of classification results/n
/n
Code execution environment:/n
python 3.5.2/n
numpy 1.16.2/n
keras 2.1.3/n
tensorflow-gpu 1.4/n
opencv-python/n
matplotlib/n
pandas/n
seaborn/n
sklearn/n
