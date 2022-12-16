## Insure-cow
Image segmentation and Explanation for Insure-cow


# Motivation:
Animal identification is vital in in large group of animals 
Individual identification allow management of 
Stockbreeding programs 
Disease and treatment 
Cattle identification
The real-time cattle recognition system provides an efficient way to stop the cattle manipulations
it is tough to verify and recognize the registered insurance animals (owner of cattle) or impostor (non-insurance) animal
Classic methods for cattle's are considered as scars and also vulnerable to manipulation
# Problem statement:
Classical cattle identification and tracking methods performance is limited due to their vulnerability to losses, duplications, fraud, and security challenges.
Investigation of bio and physical metrics that lead to best identification results (Example of these features are eyes, iris, face, weight) 
Data Sets Collection 
Cattle identification has no available data sets of cattle’s 
Suitable feature extraction for the Machine Learning algorithms


# Run our Code

First gather masks using our ``` python GUI_interface.py ```and then run the following commands:

Increase the dataset using our GAN
```python GAN image generation.py```


Run Ensemble model to get our SSIM and MSE and segmented images.

```python VGG19+KNN+ensemble.py```

