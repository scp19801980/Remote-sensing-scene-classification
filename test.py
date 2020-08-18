import os 
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from models.model import LCNN_BFF
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
#model:Model with weights imported
#labels:A list. The list includes all the category names of the dataset being tested
#img_path:The path of an image
#testset_dir:Test set path
def img_test(img_path, model, labels):
    img_array = load_img(img_path, target_size=(256, 256))
    img_array = [img_to_array(img_array)]
    x_test = np.array(img_array, dtype='float') / 255.0
    test_pred = np.argmax(model.predict(x_test), axis=1)
    score = np.amax(model.predict(x_test), axis=1)
    plt.title('Result:%s \nConfidence：%s' % (labels[test_pred[0]], score[0]))
    plt.imshow(x_test[0])
    plt.show()


def random_img_test(model, labels, testset_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        testset_dir,
        target_size=(256, 256),
        batch_size=4,
        class_mode='categorical')
    x_test, y_test = test_generator.__getitem__(0)
    preds = model.predict(x_test)

    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title('Result:%s , Real class name:%s \nConfidence：%s' % (labels[np.argmax(preds[i])], labels[np.argmax(y_test[i])], np.amax(preds[i])))
        plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
        plt.imshow(x_test[i])
    plt.show()
    
def heatmap(img_path, model):
    img_array = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img_array)
    img_array = [img]
    x_test = np.array(img_array, dtype='float') / 255.0
    test_pred = np.argmax(model.predict(x_test), axis=1)
    index = test_pred[0]
    output = model.output[:, index]
    last_conv_layer = model.get_layer(index=-3)#The last layer of convolution
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x_test])
    #512 refers to the number of channels in the last layer of convolution
    for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap *0.5 + img
    cv2.imwrite('./heatmap.jpg', superimposed_img)
    array = plt.imread('./heatmap.jpg')
    plt.imshow(array)
    plt.colorbar()
    plt.show()
