from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math

print(tf.__version__)

#batch_size = 128
IMG_HEIGHT = 28
IMG_WIDTH = 28

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trowser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print('train_images shape: ', train_images.shape)
print('train_labels shape: ', len(train_labels))
print('test_images shape: ', test_images.shape)
print('test_labels shape: ', len(test_labels))

train_images = train_images.reshape(60000, IMG_WIDTH, IMG_HEIGHT, 1)
test_images = test_images.reshape(10000, IMG_WIDTH, IMG_HEIGHT, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

validation_images = test_images[:5000]
validation_labels = test_labels[:5000]

test_images = test_images[5000:]
test_labels = test_labels[5000:]

#validation_images = train_images[55000:]
#validation_labels = train_labels[55000:]

#train_images = train_images[0:55000]
#train_labels = train_labels[0:55000]

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        img = img.reshape(IMG_WIDTH,IMG_HEIGHT)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

image_gen_train = ImageDataGenerator(rotation_range=45,
                                     width_shift_range=.15,
                                     height_shift_range=.15,
                                     horizontal_flip=True,
                                     zoom_range=0.5)
validation_image_generator = ImageDataGenerator() # Generator for our validation data


#image_gen_train.fit(train_images)                                   # adds generated images to the train_images 
train_data_gen = image_gen_train.flow(train_images, train_labels)
#validation_data_gen = validation_image_generator.flow(validation_images, validation_labels)  


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

our_optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=our_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.summary()

#history = model.fit(train_data_gen, steps_per_epoch = 60000 // batch_size, epochs = 10, validation_data=validation_data_gen, max_queue_size=50, workers=4, use_multiprocessing=False)

def scheduler(epoch):
    initial_learning_rate = 0.001
    epochs_drop = 15
    if epoch % 15 == 0:
        drop = 0.1
        learning_rate = initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        print('\n\n current learning rate: ', learning_rate, '\n\n')
        return learning_rate
    else:
        return initial_learning_rate

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

train_acc = []
train_val_acc = []
train_loss = []
train_val_loss = []

def appendHistoryValues(history):
    for item in history.history['accuracy']:
        train_acc.append(item)
    for item in history.history['val_accuracy']:
        train_val_acc.append(item)
    for item in history.history['loss']:
        train_loss.append(item)
    for item in history.history['val_loss']:
        train_val_loss.append(item)

num_epochs = 10

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
appendHistoryValues(history)

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

model.save("model1_weights.h5")

epochs_range = range(num_epochs*1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, train_val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, train_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)


    


