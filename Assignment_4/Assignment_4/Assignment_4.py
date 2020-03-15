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

print(tf.__version__)

#batch_size = 128
#epochs = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trowser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

print('train_images shape: ', train_images.shape)
print('train_labels shape: ', len(train_labels))
print('train_images shape: ', test_images.shape)
print('train_labels shape: ', len(test_labels))

train_images = train_images.reshape(60000, IMG_WIDTH, IMG_HEIGHT, 1)
test_images = test_images.reshape(10000, IMG_WIDTH, IMG_HEIGHT, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

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
validation_image_generator = keras.preprocessing.image.ImageDataGenerator() # Generator for our validation data


#image_gen_train.fit(train_images)                                   # adds generated images to the train_images 
train_data_gen = image_gen_train.flow(train_images, train_labels)
validation_data_gen = validation_image_generator.flow(test_images, test_labels)  


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.summary()

#history = model.fit(train_data_gen, steps_per_epoch = 60000 // batch_size, epochs = 10, validation_data=validation_data_gen, max_queue_size=50, workers=4, use_multiprocessing=False)

history = model.fit(train_data_gen, epochs = 10, validation_data=validation_data_gen)
history = model.fit(train_images, train_labels, validation_data=(test_images,test_labels), epochs=10)
#history = model.fit(train_data_gen, steps_per_epoch = 60000 // batch_size, epochs = 10, validation_data=validation_data_gen)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)


    


