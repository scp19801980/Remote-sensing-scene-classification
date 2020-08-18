import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
from models.model import LCNN_BFF

input_shape = (256, 256, 3)
num_classes = 45
#Fill in the number of categories of the selected dataset, for example, fill in 45 for NWPU dataset
testset_dir = 'data/test/XXX'
weight_path = 'XXX.h5'
batch_size = 64
model = LCNN_BFF(input_shape, num_classes)
model.load_weights(weight_path)

# Prediction on test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False)

for i in range(len(test_generator)):
    x_test, y_test = test_generator.__getitem__(i)
    test_true = np.argmax(y_test, axis=1)
    test_pred = np.argmax(model.predict(x_test), axis=1)
    dataframe = pd.DataFrame({'true_labels':test_true, 'pred_labels':test_pred}, columns=['true_labels', 'pred_labels'])
    if i == 0:
        dataframe.to_csv('predict.csv', sep=',', mode='w', index=False)
    else:
        dataframe.to_csv('predict.csv', sep=',', mode='a', index=False, header=False)
