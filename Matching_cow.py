# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:45:07 2021

@author: USER
"""
# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()
    
 
    
    
# # load the images -- the original, the original + contrast,
# # and the original + photoshop
# original = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/cowcrop.png")
# contrast = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/Mask.png")
# shopped = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/cow_segmented_pRF.png")
# # convert the images to grayscale
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
# shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)


#we check our input train and mask image thats why we choose the value
#Resizing images is optional, CNNs are ok with large images
SIZE_X = 280 #Resize images (height  = X, width = Y)
SIZE_Y = 220

#Test on a different image
#READ EXTERNAL IMAGE...
org_img = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/cow1.png", cv2.IMREAD_COLOR)       
org_img = cv2.resize(org_img, (SIZE_Y, SIZE_X))
original = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

        
 #Test on a different image
#READ EXTERNAL IMAGE...
mask_img = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/cow2.png", cv2.IMREAD_COLOR)       
mask_img = cv2.resize(mask_img, (SIZE_Y, SIZE_X))
contrast = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)       


shopped = cv2.imread(r"C:\Users\USER\.spyder-py3\insecurecow\cownew\check/cow_segmented_pRF.png")

shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Mask", original), ("Original", contrast), ("Predicted", shopped)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, contrast, "Original vs. Contrast")
compare_images(original, shopped, "Original vs. Photoshopped")