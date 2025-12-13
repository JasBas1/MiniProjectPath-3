#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import copy
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
  selected_images = []
  selected_labels = []
  for num in number_list:
    idx = np.where(labels == num)[0][0]  # take the first image for each class
    selected_images.append(images[idx])
    selected_labels.append(num)
  return np.array(selected_images), np.array(selected_labels)

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title.
  n = len(images)
  fig, axs = plt.subplots(1, n, figsize=(n*2, 2))
  if n == 1:
    axs = [axs]
  for i in range(n):
    axs[i].imshow(images[i], cmap='gray')
    axs[i].set_title(str(labels[i]))
    axs[i].axis('off')
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  correct = np.sum(results == actual_values)
  total = len(actual_values)
  Accuracy = correct / total
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)


#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))


#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0, max_iter=1000)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))



#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
# Note: for poisoned, scale the data
X_train_poison_flat = X_train_poison.reshape(X_train_poison.shape[0], -1)
scaler_poison = MinMaxScaler()
X_train_poison_scaled = scaler_poison.fit_transform(X_train_poison_flat)
X_test_scaled_poison = scaler_poison.transform(X_test_reshaped)

model_1_p = GaussianNB()
model_1_p.fit(X_train_poison_scaled, y_train)
model1_p_results = model_1_p.predict(X_test_scaled_poison)
Model1_p_Accuracy = OverallAccuracy(model1_p_results, y_test)
print("The overall results of the poisoned Gaussian model is " + str(Model1_p_Accuracy))

model_2_p = KNeighborsClassifier(n_neighbors=10)
model_2_p.fit(X_train_poison_scaled, y_train)
model2_p_results = model_2_p.predict(X_test_scaled_poison)
Model2_p_Accuracy = OverallAccuracy(model2_p_results, y_test)
print("The overall results of the poisoned KNN model is " + str(Model2_p_Accuracy))

model_3_p = MLPClassifier(random_state=0, max_iter=1000)
model_3_p.fit(X_train_poison_scaled, y_train)
model3_p_results = model_3_p.predict(X_test_scaled_poison)
Model3_p_Accuracy = OverallAccuracy(model3_p_results, y_test)
print("The overall results of the poisoned MLP model is " + str(Model3_p_Accuracy))



#Part 12-13
# Denoise the poisoned training data, X_train_poison.
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data.
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

X_train_poison_flat = X_train_poison.reshape(X_train_poison.shape[0], -1)
scaler_kpca = MinMaxScaler()
X_train_poison_scaled = scaler_kpca.fit_transform(X_train_poison_flat)

pca = KernelPCA(
    n_components=32,
    kernel="rbf",
    gamma=0.01,
    fit_inverse_transform=True,
    alpha=0.001,
    random_state=0,
)
pca.fit(X_train_poison_scaled)

X_denoised_scaled = pca.inverse_transform(pca.transform(X_train_poison_scaled))
X_denoised_flat = scaler_kpca.inverse_transform(X_denoised_scaled)
X_train_denoised = X_denoised_flat.reshape(X_train_poison.shape)


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
X_train_denoised_flat = X_train_denoised.reshape(X_train_denoised.shape[0], -1)
scaler_denoised = MinMaxScaler()
X_train_d_scaled = scaler_denoised.fit_transform(X_train_denoised_flat)
X_test_scaled_denoised = scaler_denoised.transform(X_test_reshaped)

model_1_d = GaussianNB()
model_1_d.fit(X_train_d_scaled, y_train)
model1_d_results = model_1_d.predict(X_test_scaled_denoised)
Model1_d_Accuracy = OverallAccuracy(model1_d_results, y_test)
print("The overall results of the denoised Gaussian model is " + str(Model1_d_Accuracy))

model_2_d = KNeighborsClassifier(n_neighbors=10)
model_2_d.fit(X_train_d_scaled, y_train)
model2_d_results = model_2_d.predict(X_test_scaled_denoised)
Model2_d_Accuracy = OverallAccuracy(model2_d_results, y_test)
print("The overall results of the denoised KNN model is " + str(Model2_d_Accuracy))

model_3_d = MLPClassifier(random_state=0, max_iter=1000)
model_3_d.fit(X_train_d_scaled, y_train)
model3_d_results = model_3_d.predict(X_test_scaled_denoised)
Model3_d_Accuracy = OverallAccuracy(model3_d_results, y_test)
print("The overall results of the denoised MLP model is " + str(Model3_d_Accuracy))
