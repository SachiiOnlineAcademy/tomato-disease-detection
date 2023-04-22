import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# dataset Path

train = r'C:\Users\user\PycharmProjects\NewDataset\train'
test = r'C:\Users\user\PycharmProjects\NewDataset\test'

trainDIR = r'C:\Users\user\PycharmProjects\NewDataset\train'
testDIR = r'C:\Users\user\PycharmProjects\NewDataset\test'

# data Size

size = 256
batch_size = 64
epoch = 20

# Data Augmentation

datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             validation_split=0.2)

X_train = datagen.flow_from_directory(train, target_size=(size, size), batch_size=batch_size, class_mode='categorical',
                                      subset='training')

X_test = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test, target_size=(size, size), batch_size=batch_size,
                                                                  class_mode='categorical', subset='training')

X_test.class_indices.keys()

# call back setup

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint = ModelCheckpoint(r'test-model.h5', monitor='val_loss', mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0,
                          patience=20, verbose=1, restore_best_weights=True)

callbacks = [checkpoint, earlystop]

# CNN Model

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=(size, size, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# print model summary

history = model.fit(x=X_train, validation_data=X_test, epochs=epoch, steps_per_epoch=X_train.samples // batch_size,
                    validation_steps=X_test.samples // batch_size, callbacks=callbacks)

# save model
model.save("C:/Users/user/PycharmProjects/pythonModelGUI/newnew_trained_model.h5")
# history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, 21)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
