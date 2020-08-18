from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, Input, BatchNormalization, \
     add, Activation
from keras.regularizers import l2
from keras.models import Model
from .depthwise_conv2d import DepthwiseConv2D

def LCNN_BFF(input_shape, num_classes):
    input_0 = Input(shape=input_shape)
    # Group 1 256*256
    x = Conv2D(32, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(input_0)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(32, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Group 2 128*128
    x = Conv2D(64, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(64, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Group 3 64*64
    x = Conv2D(128, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(128, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    a = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Group 4 32*32
    x = Conv2D(128, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(a)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(128, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    q = BatchNormalization()(x)


    x = Conv2D(128, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(a)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(128, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    w = BatchNormalization()(x)

    x = add([q, w])
    e = Activation('relu')(x)

    # Group 5 16*16
    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    r = BatchNormalization()(x)

    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    t = BatchNormalization()(x)

    x = add([r, t])
    e = Activation('relu')(x)

    # Group 6 8*8
    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    q = BatchNormalization()(x)

    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=2, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    w = BatchNormalization()(x)

    x = add([q, w])
    e = Activation('relu')(x)

    # Group 7 4*4
    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    r = BatchNormalization()(x)

    x = Conv2D(256, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(256, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    t = BatchNormalization()(x)

    x = add([r, t])
    e = Activation('relu')(x)

    # Group 8 4*4
    x = Conv2D(512, (1, 1), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(e)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(512, (3, 3), padding='same', strides=1, kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Group 9 full connection
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input_0, x)
    return model