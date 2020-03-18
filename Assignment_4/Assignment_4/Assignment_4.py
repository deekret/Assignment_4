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
import models
import imp
imp.reload(models)

print(tf.__version__)


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

train_data_gen = image_gen_train.flow(train_images, train_labels)
#validation_data_gen = validation_image_generator.flow(validation_images, validation_labels)  


#model = models.createModel1()
model = models.createModel2()
#model = models.createModel3()

step = 0
def scheduler(epoch, lr):
    if (((epoch + step) % 15 == 0) and (epoch + step) != 0):
        lr = lr * 0.1
    print("epoch: ", epoch + step)
    print("lr : ", lr)
    return lr

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

num_epochs = 1
print("read the learning rate ---> ", keras.backend.eval(model.optimizer.lr))

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
appendHistoryValues(history)
step += num_epochs

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)

#history = model.fit(train_data_gen, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)
#step += 10

#history = model.fit(train_images, train_labels, callbacks=[callback], validation_data=(validation_images,validation_labels), epochs = num_epochs)
#appendHistoryValues(history)
#step += num_epochs

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
print("predictions[0]", predictions[0])
predictions = [np.argmax(prediction) for prediction in predictions]
print("predictions[0] after", predictions[0])

print("shape of predictions: ", len(predictions))
print("validation shape: ", validation_labels.shape)
confusion_matrix = tf.math.confusion_matrix(validation_labels, predictions)
print("confusion matrix: ", confusion_matrix)

    


