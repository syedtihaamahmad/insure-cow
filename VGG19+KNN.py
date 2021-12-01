# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 23:26:06 2021

@author: USER
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#crack_2 = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\Prep2_Cropped_670_658/frame_0174-Crop2.jpg")
#:\Users\USER\.spyder-py3\insecurecow\Prep2_Cropped_670_658\crack_new-train
print(os.listdir(r"C:\Users\USER\.spyder-py3\insecurecow/cownew/"))


#we check our input train and mask image thats why we choose the value
#Resizing images is optional, CNNs are ok with large images
SIZE_X = 280 #Resize images (height  = X, width = Y)
SIZE_Y = 220

#Capture training image info as a list
train_images = []

for directory_path in glob.glob(r"C:\Users\USER\.spyder-py3\insecurecow\cownew/train/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
        
        
        
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob(r"C:\Users\USER\.spyder-py3\insecurecow\cownew/mask/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.PNG")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 


#Load VGG19 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
VGG_model = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()

#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)

#Plot features to view them
square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()

#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset['Label'] != 0]

#Redefine X and Y for KNN
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']







########## This part belongs only the testing the model accuracy ########

#Define the dependent variable that needs to be predicted (labels)
Y =  dataset["Label"].values

#Define the independent variables
X =  dataset.drop(labels= ["Label"], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

# model = KNeighborsClassifier(n_neighbors=6)


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, random_state = 42)



# from sklearn.svm import LinearSVC
# model = LinearSVC(max_iter=100)  #Default of 100 is not converging


# Train the model on training data
model.fit(X_train, y_train)


#First test prediction on the training data itself. SHould be good. 
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)


#Let us check the accuracy on test data
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy of testing data = ", metrics.accuracy_score(y_test, prediction_test))


####### End the part of Accuracy ###########

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=6)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, random_state = 42)



# from sklearn.svm import LinearSVC
# model = LinearSVC(max_iter=100)  #Default of 100 is not converging



######### Now time to test the image using the  model so we no use the train and test split data######


#Redefine X and Y for KNN
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']


# Train the model on training data
model.fit(X_for_RF, Y_for_RF) 

#Save model for future use
filename = 'RF_model.sav'
pickle.dump(model, open(filename, 'wb'))

#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))


#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread(r'C:\Users\USER\.spyder-py3\insecurecow\cownew/cow223.jpeg', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)

#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = new_model.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

prediction = loaded_model.predict(X_test_feature)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='binary')
plt.imsave(r'C:\Users\USER\.spyder-py3\insecurecow\cownew/cow_segmented_RF_another.jpg', prediction_image, cmap='gray')
