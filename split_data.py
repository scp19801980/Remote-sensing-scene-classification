import os
import random
import shutil
train_percent = 0.8

original_data_path = 'original_data/NWPU45/'
data = os.listdir(original_data_path)
# Next, all kinds of image paths are written into the dictionary to facilitate subsequent reading
image_path = {}
for class_name in data:
    path_a = original_data_path + class_name + '/'
    image_file = os.listdir(path_a)
    x = []
    for name in image_file:
        image_filepath = path_a + name
        x.append(image_filepath)
    image_path[class_name] = x
# Random sampling data, and then copy the image to the target path
for class_name in data:
    image = image_path[class_name]
    train_data = random.sample(image, int(len(image)*train_percent))
    test_data = []
    for i in image:
        if i not in train_data:
            test_data.append(i)
    # Copy the training set
    for j in train_data:
        save_path_1 = 'data/train/' + j[14:] #  14 refers to the length of the character 'original_data/'
        save_path_2 = 'data/train/new_data/' + class_name + '/' #Judge whether the path exists or not. If it does not exist, it will be created
        if not os.path.exists(save_path_2):
            os.makedirs(save_path_2)
        shutil.copy(j, save_path_1)
    # Copy the test set
    for k in test_data:
        save_path_1 = 'data/test/' + k[14:] #  14 refers to the length of the character 'original_data/'
        save_path_2 = 'data/test/new_data/' + class_name + '/' #Judge whether the path exists or not. If it does not exist, it will be created
        if not os.path.exists(save_path_2):
            os.makedirs(save_path_2)
        shutil.copy(k, save_path_1)