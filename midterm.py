import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#Exercise 1: Classify images of handwritten digits

#create the input node
inputs=keras.Input(shape=(784,))
#input shape of the image
img_imputs=keras.Input(shape=(28,28,1))

#add a new node in the graph of layers
#first hidden layer
hidden1=layers.Dense(56,activation='relu')(inputs)
#second hidden layer
hidden2=layers.Dense(56,activation='relu')(hidden1)
#output
outputs=layers.Dense(10,activation='softmax')(hidden2)
#define the model
model=keras.Model(inputs=inputs,outputs=outputs,name='mnist_grayscale')
#summary
model.summary()
#plot of the model
keras.utils.plot_model(model,show_shapes=True)

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


x_train_norm=x_train.reshape(60000,784).astype("float32")/255
x_test_norm=x_test.reshape(10000,784).astype("float32")/255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"])

epoch_number=10
history=model.fit(x_train_norm,y_train,batch_size=56,epochs=epoch_number,validation_split=0.2)

fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].plot(range(epoch_number),history.history['accuracy'],label='Training')
axs[0].plot(range(epoch_number),history.history['val_accuracy'],label='Testing')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Model accuracy')
axs[1].plot(range(epoch_number),history.history['loss'],label='Training')
axs[1].plot(range(epoch_number),history.history['val_loss'],label='Testing')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Model loss')
plt.show()

#plotting individual numbers
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))

