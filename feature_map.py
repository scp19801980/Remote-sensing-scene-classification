import numpy as np
import os
import matplotlib.pyplot as plt
# import cv2
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from models.model import LCNN_BFF

input_shape = (256, 256, 3)
num_classes = 45
batch_size = 64
data_path = 'original_data/NWPU45/'
class_names = os.listdir(data_path)
# print(class_names)
image_paths = []
for c_name in class_names:
    class_path = data_path + c_name + '/'
    image_name = os.listdir(class_path)
    for i in range(len(image_name)):
        image_name[i] = class_path + image_name[i]
        image_paths.append(image_name[i])
# print(image_paths)
for c_name in class_names:
    save = 'feature_map/BFF/' + c_name + '/'
    if not os.path.exists(save):
        os.makedirs(save)

weights_path = '.h5'
model = LCNN_BFF(input_shape, num_classes)
model.load_weights(weights_path)
#The fourth group, the first branch-36, the second branch-37, bff-38, channel = 128
conv_layer = Model(inputs=model.inputs, outputs=model.get_layer(index=38).output)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    data_path,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False)

c = 0
for i in range(len(test_generator)):
    x_test, y_test = test_generator.__getitem__(i)
    conv_output = conv_layer.predict(x_test)
    for j in range(batch_size):
        total_feature_map = conv_output[j, :, :, 0]
        for k in range(1, 128):
            single_feature_maps = conv_output[j, :, :, k]
            total_feature_map = total_feature_map + single_feature_maps

        plt.figure(num=1, figsize=(2, 1.5), dpi=60, clear=True)
        plt.imshow(total_feature_map)

        save_path = 'feature_map/BFF/' + image_paths[c][21:-3] + 'png'
        plt.savefig(save_path)
        c = c + 1


