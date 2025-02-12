import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.utils.visualize_util import plot
import pydot
import graphviz

IMG_HEIGHT = 28
IMG_WIDTH = 28

def createModel1():
    name = "model_1"
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
    #keras.utils.plot_model(model, 'model1.png', show_shapes=True)
    return model, name

def conv_block(x):
    #x = keras.layers.BatchNormalization()
    x = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    return x


def createModel2():
    name = "model_2"
    inputs = tf.keras.Input(shape=(IMG_WIDTH,IMG_HEIGHT,1))
    x = keras.layers.Conv2D(64, kernel_size=5, padding='same',strides = (2,2), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x_residual = x
    x_dense = x
    x = keras.layers.Conv2D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu')(x)

    x = keras.layers.add([x, x_residual])

    for i in range(4):
        cb = conv_block(x)
        x = keras.layers.Concatenate()([x, cb])

    x = keras.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    our_optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=our_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()
    return model, name

def createModel3():
    name = "model_3"
    inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(inputs)
    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    block_1_output = keras.layers.MaxPooling2D(3)(x)

    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(block_1_output)
    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    block_2_output = keras.layers.add([x, block_1_output])

    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(block_2_output)
    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    block_3_output = keras.layers.add([x, block_2_output])

    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(block_3_output)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs, outputs)

    our_optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=our_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()
    return model, name