import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Exercise 1: Classify images of handwritten digits

#get the data
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train_norm=x_train.reshape(60000,784).astype("float32")/255
x_test_norm=x_test.reshape(10000,784).astype("float32")/255

###############Architecture A of a neural network without convolution

#create the input node
inputs=keras.Input(shape=(784,))


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
#illustrate with diagram using function in keras 
keras.utils.plot_model(model,show_shapes=True)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"])

#set the epoch number
epoch_number=10
#train the neural network on the mnist dataset (training set only)
history=model.fit(x_train_norm,y_train,batch_size=56,epochs=epoch_number,validation_split=0.2)

fig,axs=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('NN Model training and loss')
axs[0].plot(range(epoch_number),history.history['accuracy'],label='Training')
axs[0].plot(range(epoch_number),history.history['val_accuracy'],label='Testing')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Model accuracy')
axs[0].legend()
axs[1].plot(range(epoch_number),history.history['loss'],label='Training')
axs[1].plot(range(epoch_number),history.history['val_loss'],label='Testing')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Model loss')
axs[1].legend()
plt.legend()
plt.show()

#choose an index 
index_of_value_to_predict=1
#get predictions on testing data
y_hat_test=model.predict(x_test_norm)
#plot the number from the testing set 
plt.imshow(x_test[index_of_value_to_predict],cmap=plt.get_cmap('gray'))
#prediction given by the model
prediction_1=np.where(y_hat_test[index_of_value_to_predict,:]==max(y_hat_test[index_of_value_to_predict,:]))[0][0]
print('\nThe true label of the value in the testing set is ',y_test[index_of_value_to_predict],'\nThe predicted value is ',prediction_1)

###############Architecture B of a convolutional neural network
#change shape of data for convolution



x_train_dims=np.expand_dims(x_train/255,axis=-1)
x_test_dims=np.expand_dims(x_test/255,axis=-1)
#model 

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#illustrate with diagram using function in keras 
keras.utils.plot_model(model,show_shapes=True)
#fit the model
history_cnn = model.fit(x_train_dims, y_train, epochs=10,validation_split=0.2 )
#plot training and validation loss and accuracy
fig,axs=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('CNN Model training and loss')
axs[0].plot(range(epoch_number),history_cnn.history['accuracy'],label='Training')
axs[0].plot(range(epoch_number),history_cnn.history['val_accuracy'],label='Testing')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Model accuracy')
axs[0].legend()
axs[1].plot(range(epoch_number),history_cnn.history['loss'],label='Training')
axs[1].plot(range(epoch_number),history_cnn.history['val_loss'],label='Testing')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Model loss')
axs[1].legend()
plt.show()



