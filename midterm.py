import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#Exercise 1: Classify images of handwritten digits

#create the input node
inputs=keras.Input(shape=(784,))
#input shape of the image
img_imputs=keras.Input(shape=(28,28,1))

#add a new node in the graph of layers
dense=layers.Dense(56,activation='relu')
x=dense(inputs)

x=layers.Dense(64,activation='relu')(x)
outputs=layers.Dense(10,activation='softmax')(x)

model=keras.Model(inputs=inputs,outputs=outputs,name='mnist_grayscale')
