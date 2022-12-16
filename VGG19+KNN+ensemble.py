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
from keras.layers import Conv2D, Flatten
import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import shap



#we check our input train and mask image thats why we choose the value
#Resizing images is optional, CNNs are ok with large images
SIZE_X = 224 #Resize images (height  = X, width = Y)280 
SIZE_Y = 224 #220

#Capture training image info as a list
train_images = []

for directory_path in glob.glob(r"./train/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
        
        
        
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

x_expl_train=np.reshape(train_images,(len(train_images),SIZE_X,SIZE_Y,3))
#print(x_expl_train.shape)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob(r"./mask/"):
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
VGG_model2 = VGG19(weights='imagenet',include_top=False, input_shape=(SIZE_X, SIZE_Y, 3),classes=2)
"""
VGG_model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
VGG_model.fit(X_train,y_train)
"""
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
#new_model_exp = Model(inputs=VGG_model2.input, outputs=VGG_model2.get_layer('block5_conv4').output)
#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)

from keras.layers import Input, Dense, concatenate
#model_b=Sequential()
#for i in new_model.layers[:2]: model_b.add(i)
#bl_1 =model_b.add(Flatten())


#output = model_b(new_model.outputs)
#joinedModel = Model(inputs=VGG_model.input)

#print(model_b(VGG_model.input))




#shap.initjs()
#shap.force_plot(e.expected_value[1], shap_values[1], X_test[0],show=True)

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

#model = KNeighborsClassifier(n_neighbors=6)
print(X_test.shape)
print(X_train.shape)

model_a=Sequential()
model_a.add(Flatten())
model_a.add(Dense(64,activation="relu"))
model_a.add(Dense(1))
model_a.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_a.fit(X_train, y_train)

"""
topk = 4
batch_size = 50
n_evals = 10000

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(128,128)", X_train[0].shape)

# create an explainer with model and image masker
explainer = shap.Explainer(model_a, masker_blur)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(X_train[1:2], max_evals=n_evals, batch_size=batch_size,
                        outputs=shap.Explanation.argsort.flip[:topk])



background = X_train[:3]
print(background.shape)
#print(VGG_model.predict(X_train).shape)
# explain predictions of the model on four images
explainer = shap.DeepExplainer(VGG_model,background)
shap_values = explainer.shap_values(X_train[:1])
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
#shap_values = e.shap_values(X_test[0])
print(shap_values)
"""
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators = 100, random_state = 42)
with open('saved_models/model1.pkl', 'wb') as file:
    pickle.dump(model1, file)




# Train the model on training data
model1.fit(X_train, y_train)


#First test prediction on the training data itself. SHould be good. 
prediction_test_train = model1.predict(X_train)

#Test prediction on testing data. 
prediction_test = model1.predict(X_test)


#Let us check the accuracy on test data
#Print the prediction accuracy

#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy of testing data = ", metrics.accuracy_score(y_test, prediction_test))
""""
from sklearn.svm import LinearSVC
model3 = LinearSVC(max_iter=150)  #Default of 100 is not converging
model3.fit(X_train, y_train)
with open('saved_models/model3.pkl', 'wb') as file:
    pickle.dump(model3, file)
"""

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=6)
model2.fit(X_train, y_train)
with open('saved_models/model2.pkl', 'wb') as file:
    pickle.dump(model2, file)


from sklearn.neural_network import MLPClassifier
model3 = MLPClassifier()  #Default of 100 is not converging
model3.fit(X_train, y_train)
with open('saved_models/model3.pkl', 'wb') as file:
    pickle.dump(model3, file)
### Model average / sum Ensemble
# Simple sum of all outputs / predictions and argmax across all classes
########
#from keras.models import load_model
from sklearn.metrics import accuracy_score

models = [model1, model2, model3]
preds = [model.predict(X_test) for model in models]
preds=np.array(preds)
print(preds)

summed = np.sum(preds, axis=0)
# argmax across classes
ensemble_prediction = np.divide(summed, len(models))
ensemble_prediction=ensemble_prediction.astype(int)
#ensemble_prediction=summed
print(ensemble_prediction)
prediction1 = model1.predict(X_test)
prediction2 = model2.predict(X_test)
prediction3 = model3.predict(X_test)


accuracy1 = accuracy_score(y_test, prediction1)
accuracy2 = accuracy_score(y_test, prediction2)
accuracy3 = accuracy_score(y_test, prediction3)
ensemble_accuracy = accuracy_score(y_test, ensemble_prediction)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)

########################################
#Weighted average ensemble
models = [model1, model2, model3]
preds = [model.predict(X_test) for model in models]
preds=np.array(preds)
weights = [0.4, 0.4, 0.2]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
#weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)
ensemble_prediction = np.divide(weighted_preds, 1)
weighted_ensemble_prediction=ensemble_prediction.astype(int)

weighted_accuracy = accuracy_score(y_test, weighted_ensemble_prediction)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)
print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)

########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy
models = [model1, model2, model3]
preds1 = [model.predict(X_test) for model in models]
preds1=np.array(preds1)
print(preds1)
import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 1):
    for w2 in range(0,1):
        for w3 in range(0,1):
            wts = [w1,w2,w3]
            wted_preds1 = np.tensordot(preds1, wts, axes=((0),(0)))
            #wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
            ensemble_prediction = np.divide(wted_preds1, 1)
            wted_ensemble_pred=ensemble_prediction.astype(int)
            weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'acc':weighted_accuracy*100}, index=[0]), ignore_index=True)
            
max_acc_row = df.iloc[df['acc'].idxmax()]
print("Max accuracy of ", max_acc_row[0], " obained with w1=", max_acc_row[1],
      " w2=", max_acc_row[2], " and w3=", max_acc_row[3])         




###########################################################################
### Explore metrics for the ideal weighted ensemble model. 

models = [model1, model2, model3]
preds = [model.predict(X_test) for model in models]
preds=np.array(preds)
ideal_weights = [0.4, 0.4,0.2] 

#Use tensordot to sum the products of all elements over specified axes.
ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
#ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)
ensemble_prediction = np.divide(ideal_weighted_preds, 1)
ideal_weighted_ensemble_prediction=ensemble_prediction.astype(int)
ideal_weighted_accuracy = accuracy_score(y_test, ideal_weighted_ensemble_prediction)
print("Ideal weighted score:",ideal_weighted_accuracy)


#RANDOM FOREST
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators = 200, random_state = 42)

#x_expl_train=np.reshape(X_train,(64,SIZE_X,SIZE_Y,1))
# select a set of background examples to take an expectation over
"""
background = x_expl_train

# explain predictions of the model on four images
e = shap.TreeExplainer(model1)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(X_test[0])
print(shap_values)
shap.initjs()
shap.force_plot(e.expected_value[1], shap_values[1], X_test[0],show=True)
"""



######### Now time to test the image using the  model so we no use the train and test split data######


#Redefine X and Y for KNN
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']


# Train the model on training data
model1.fit(X_for_RF, Y_for_RF) 

#Save model for future use
filename = 'RF_model.sav'
pickle.dump(model1, open(filename, 'wb'))

#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))


#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread(r'./train/cow222.jpeg', cv2.IMREAD_COLOR)       
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
plt.imsave(r'cow_segmented_RF_another.jpg', prediction_image, cmap='gray')
