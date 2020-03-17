import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.utils.visualize_util import plot
import pydot
import graphviz

IMG_HEIGHT = 28
IMG_WIDTH = 28

def createModel1():
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
    keras.utils.plot_model(model, 'model1.png', show_shapes=True)
    return model

def createModel2():
    inputs = tf.keras.Input(shape=(IMG_WIDTH,IMG_HEIGHT,1))
    x = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x_shortcut = x
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.add([x, x_shortcut])

    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)

    #print("output shape ---- > ", x.shape)
    #x_shortcut = keras.layers.Reshape( (x.shape[1], x.shape[2], x.shape[3]) )(x_shortcut)
    #print("checking of shape : ", x_shortcut.shape[1], x_shortcut.shape[2], x_shortcut.shape[3])

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    our_optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=our_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()
    #keras.utils.plot_model(model, 'model2.png', show_shapes=True)
    #tf.keras.utils.visualize_util.plot(model, 'model2.png', show_shapes=True)
    return model