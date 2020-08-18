from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD
from models.model import LCNN_BFF

trainset_dir = 'data/train/NWPU45/'
valset_dir = 'data/test/NWPU45/'
num_classes = 45
learning_rate = 1e-2
momentum = 0.9
batch_size = 16
input_shape = (256, 256, 3)

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 60,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    trainset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    valset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')

optim = SGD(lr=learning_rate, momentum=momentum)
# optim = RMSprop(lr=learning_rate)
# optim = Adam(amsgrad=True)

model = LCNN_BFF(input_shape, num_classes)

model.compile(optimizer=optim, loss='categorical_crossentropy',
              metrics=['acc'])

csv_path = 'result/XXX.csv'
save_weights_path = 'model-weight-ep-{epoch:02d}-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.h5'
#You can modify the path by yourself

checkpoint = ModelCheckpoint(save_weights_path, monitor='val_acc', verbose=1, 
                             save_weights_only=True, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=32, verbose=1)
# logging = TensorBoard(log_dir=log_dir, batch_size=batch_size)
csvlogger = CSVLogger(csv_path, append=True)

callbacks = [checkpoint, reduce_lr, csvlogger]

num_epochs = 1000

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=num_epochs,
                    verbose=1, 
                    callbacks=callbacks, 
                    validation_data=val_generator, 
                    validation_steps=len(val_generator),
                    workers=1)
# fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, 
#               callbacks=None, validation_data=None, validation_steps=None, 
#               class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)