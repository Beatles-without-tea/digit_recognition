import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import pandas as pd

#####################################################
#Exercise 1: Classify images of handwritten digits
#####################################################

#fetch the data
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train_norm=x_train.reshape(60000,784).astype("float32")/255
x_test_norm=x_test.reshape(10000,784).astype("float32")/255

###############
###############Architecture A of a neural network without convolution
###############

#create the input node
inputs=keras.Input(shape=(784,))

#Feed forward neural network with 2 hidden layers
#first hidden layer
#activation using the rectified linear function
hidden1=layers.Dense(56,activation='relu')(inputs)
#second hidden layer
#activation using the rectified linear function
hidden2=layers.Dense(56,activation='relu')(hidden1)
#output
outputs=layers.Dense(10,activation='softmax')(hidden2)
#define the model
model=keras.Model(inputs=inputs,outputs=outputs,name='mnist_grayscale')
#summary
model.summary()
#illustrate with diagram using function in keras 
keras.utils.plot_model(model,show_shapes=True)
#compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='sgd',
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

#prediction given by the model
prediction_NN=np.where(y_hat_test[index_of_value_to_predict,:]==max(y_hat_test[index_of_value_to_predict,:]))[0][0]
print('\nThe true label of the value in the testing set is ',y_test[index_of_value_to_predict],'\nThe predicted value is ',prediction_NN,' with probability: ',max(y_hat_test[index_of_value_to_predict,:]))

#View network predictions (probabilities of class membership)
#plot the number from the testing set of the given index we want to predict
fig,axs=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('NN predictions')
axs[0].imshow(x_test[index_of_value_to_predict],cmap=plt.get_cmap('gray'))
axs[0].set_title('True value')
axs[1].bar(range(0,10),list(y_hat_test[index_of_value_to_predict,:]),label='The NN predicts '+str(prediction_NN)+' with probability '+str(max(y_hat_test[index_of_value_to_predict,:])))
axs[1].legend(bbox_to_anchor=(1,-0.12))
axs[1].set_title('Probabilities of class membership')
axs[1].set_ylabel('probability')
axs[1].set_xlabel('number')
plt.show()

#evaluate on mnist dataset
scores_NN = model.evaluate(x_test_norm, y_test)
print('The NN model gives us an accuracy of '+str(scores_NN[1])+' and a loss of '+str(scores_NN[0]))

###############
###############Architecture B of a convolutional neural network
###############

#change shape of data for convolution
x_train_dims=np.expand_dims(x_train/255,axis=-1)
x_test_dims=np.expand_dims(x_test/255,axis=-1)

#model 
model_cnn = keras.Sequential()
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(64, activation='relu'))
model_cnn.add(layers.Dense(10,activation='softmax'))
model_cnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#illustrate with diagram using function in keras 
keras.utils.plot_model(model_cnn,show_shapes=True)
#fit the model
history_cnn = model_cnn.fit(x_train_dims, y_train, epochs=epoch_number,validation_split=0.2 )
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

#predictions
#choose an index 
index_of_value_to_predict_cnn=11
#get predictions on testing data
y_hat_test_cnn=model_cnn.predict(x_test_dims)
#prediction given by the model
prediction_cnn=np.where(y_hat_test_cnn[index_of_value_to_predict_cnn,:]==max(y_hat_test_cnn[index_of_value_to_predict_cnn,:]))[0][0]
print('\nThe true label of the value in the testing set is ',y_test[index_of_value_to_predict_cnn],'\nThe predicted value using the CNN model is ',prediction_cnn)

#View network predictions (probabilities of class membership)
fig,axs=plt.subplots(1,2,figsize=(10,5))
fig.suptitle('CNN predictions')
axs[0].imshow(x_test[index_of_value_to_predict_cnn],cmap=plt.get_cmap('gray'))
axs[0].set_title('True value')
axs[1].bar(range(0,10),list(y_hat_test_cnn[index_of_value_to_predict_cnn,:]),label='The NN predicts '+str(prediction_cnn)+' with probability '+str(max(y_hat_test[index_of_value_to_predict_cnn,:])))
axs[1].legend(bbox_to_anchor=(1,-0.12))
axs[1].set_title('Probabilities of class membership')
axs[1].set_ylabel('probability')
axs[1].set_xlabel('number')
plt.show()

scores_cnn = model_cnn.evaluate(x_test_dims, y_test)
print('\nThe CNN model gives us an accuracy of '+str(scores_cnn[1])+' and a loss of '+str(scores_cnn[0]))


#############################################################################
#Exercise 2: Detect the presence of a hand-written digit on an image
#############################################################################


#create random noise images
random_images=np.random.uniform(0,255,(30000,28 ,28))
#normalize data
random_images_norm=random_images.reshape(30000,784).astype("float32")/255
#make labels
#1 for digit, 0 for non digit
digit_labels=np.array([1 if i<=30000 else 0 for i in range(60000)])
#show random image
plt.imshow(random_images[0],cmap=plt.get_cmap('gray'))
plt.show()
#making training set for the neural network
x_train_digit=np.append(x_train_norm[:30000,:],random_images_norm,axis=0)
x_train_digit,digit_labels=shuffle(x_train_digit,digit_labels)
#create the input node
inputs=keras.Input(shape=(784,))


#Feed forward neural network with 2 hidden layers
#first hidden layer
#activation using the rectified linear function
hidden1=layers.Dense(56,activation='relu')(inputs)
#second hidden layer
#activation using the rectified linear function
hidden2=layers.Dense(56,activation='relu')(hidden1)
#output
outputs=layers.Dense(2,activation='softmax')(hidden2)
#define the model
model_digit=keras.Model(inputs=inputs,outputs=outputs,name='digit_recognition')
#summary
model_digit.summary()
#illustrate with diagram using function in keras 
keras.utils.plot_model(model_digit,show_shapes=True)


model_digit.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer='sgd',
    metrics=["accuracy"])

#set the epoch number
epoch_number=5
#train the neural network on the mnist dataset (training set only)
history_digit=model_digit.fit(x_train_digit,digit_labels,batch_size=56,epochs=epoch_number,validation_split=0.2)

#make testing data
#create random noise images
random_images_test=np.random.uniform(0,255,(10000,28 ,28))
#normalize data
random_images_norm_test=random_images_test.reshape(10000,784).astype("float32")/255
#make labels
#1 for digit, 0 for non digit
digit_labels_test=np.array([1 if i<=10000 else 0 for i in range(20000)])
#making testing set for the neural network
x_test_digit=np.append(x_test_norm[:10000,:],random_images_norm_test,axis=0)
x_test_digit,digit_labels_test=shuffle(x_test_digit,digit_labels_test)
#Evaluate the classifier’s performance using the 10, 000 MNIST test images and 10, 000 randomly
#generated images
scores = model_digit.evaluate(x_test_digit,digit_labels_test)

#Propose a vizualization of the classifier’s predictions.
#we shall be using a confusion matrix
#get predictions
predictions=model_digit.predict(x_test_digit)
y_hat_digit=np.array([0 if predictions[i][0]>predictions[i][1] else 1 for i in range(len(predictions))])
#make confusion matrix
conf_matrix=confusion_matrix(digit_labels_test, y_hat_digit)
conf_matrix_df=pd.DataFrame(columns=['Predicted 1','Predicted 0'],index=['Actual 1','Actual 0'])
conf_matrix_df.iloc[0,:]=conf_matrix[0,:]
conf_matrix_df.iloc[1,:]=conf_matrix[1,:]
print(conf_matrix_df)


############################################
#Exercise 3: Practice on other datasets
############################################


#we shall be using the fashion mnist dataset
#we import the data from tensorflow
fashion_mnist = tf.keras.datasets.fashion_mnist
#we separate into testing and training data
(x_fashion_train,y_fashion_train),(x_fashion_test,y_fashion_test)= fashion_mnist.load_data()
print('The training data is of shape: ',x_fashion_train.shape)
print('The training labels are of shape: ',y_fashion_train.shape)
print('The testing data is of shape: ',x_fashion_test.shape)
print('The testing labels are of shape: ',y_fashion_test.shape)
print('There are',len(np.unique(y_fashion_train)),'labels')
#The labels each correspond to a type of clothing:
    #0-T-shirt-top
    #1 Trouser
    #2 Pullover
    #3 Dress
    #4 Coat
    #5 Sandal
    #6 Shirt
    #7 Sneaker
    #8 Bag
    #9 Ankle boot

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#view image
plt.imshow(x_fashion_train[1],cmap=plt.get_cmap('gray'))
plt.show()
#We can see it's a t-shirt


#x_fashion_train=x_fashion_train.reshape(60000,784).astype("float32")/255
#x_fashion_train=x_fashion_test.reshape(10000,784).astype("float32")/255

x_fashion_train=np.expand_dims(x_fashion_train/255,axis=-1)
x_fashion_test=np.expand_dims(x_fashion_test/255,axis=-1)

model_fashion=keras.Sequential()
model_fashion.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_fashion.add(layers.MaxPooling2D((2, 2)))
model_fashion.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_fashion.add(layers.MaxPooling2D((2, 2)))
model_fashion.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_fashion.add(layers.Flatten())
model_fashion.add(layers.Dense(64, activation='relu'))
model_fashion.add(layers.Dense(10,activation='softmax'))
model_fashion.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#illustrate with diagram using function in keras 
keras.utils.plot_model(model_fashion,show_shapes=True)
#summary
model_fashion.summary()


#set the epoch number
epoch_number=5
#train the neural network on the mnist dataset (training set only)
history_fashion=model_fashion.fit(x_fashion_train,y_fashion_train,batch_size=56,epochs=epoch_number,validation_split=0.2)


#generated images
scores_fashion = model_fashion.evaluate(x_fashion_test,y_fashion_test)
