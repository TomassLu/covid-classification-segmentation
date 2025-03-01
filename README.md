# covid-classification-segmentation
Simple covid classification, lung and infection segmentation from image

# How to use
Execute main.py (to use another image change test_image_path in main.py)
The code will show the result and save it as an image "Detection_result.png"

# Libraries used
Tensorflow, keras, matplot, numpy, cv2, os
(Tensorflow and keraas versions - 2.13)

# Used method info
Three models were trained: densenet201 for image classification and 2 unets for lung and infection segmentation (for densenet batch size used - 16, for unets - 8, optimizer - adam)
A simple unet architecture with middle Conv(256) bridge was chosen due to faster training
The models are saved and then loaded by main python file to be used on single input image
Detection results are shown with matplot (infection segmentation only shown if predicted class is COVID)

# Data used
The data used was taken from https://www.kaggle.com/datasets/anasmohammedtahir/covidqu (only the infection segmentation data)
The lung masks were moved to "Lung Masks" folder, infection masks to - "Infection Masks" folder and the x-ray images were left in "Infection Segmentation Data" folder
All used data can be found in kagglehub folder

# Model training info
Due to time constraints the models were trained with low amount of epochs (densenet - 5, unets - 10) and as such the results are not very accurate and serve more as proof of concept


