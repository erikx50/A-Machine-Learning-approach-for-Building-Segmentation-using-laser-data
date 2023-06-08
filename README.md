# Master Thesis: A Machine Learning Approach for Building Segmentation using laser data
The code in this repository contains my solution for the **NORA MapAI: Precision in Building Segmentation** competition. More can be seen here: https://github.com/Sjyhne/MapAI-Competition

## Results:
|   Team   | IoU Task 1 | BIoU Task 1 | Total Task 1 | IoU Task 2 | BIoU Task 2 | Total Task 2 |    Score   | 
|:--------:|:------------:|:-------------:|:--------------:|:------------:|:-------------:|:--------------:|:----------:|
| Me |    0.8011    |     0.6295    |     0.7153     |    0.8964    |     0.8009    |     0.8486     | **0.7820** |
| 1st MapAI competition |    0.7794    |     0.6115    |     0.6955     |    0.8775    |     0.7857    |     0.8316     | **0.7635** |
| 2nd MapAI competition |    0.7879    |     0.6245    |     0.7062     |    0.8711    |     0.7504    |     0.8108     | **0.7585** |
| 3rd MapAI competition |    0.7902    |     0.6185    |     0.7044     |    0.8506    |     0.7461    |     0.7984     | **0.7514** |

**Task 1**
Results were achieved using an ensemble of U-Net DenseNet201, CT-UNet EfficientNetB4, and CT-UNet EfficienNetV2S with a weight of 0.4, 0.3, and 0.3.

**Task 2**
Results were achieved using an ensemble of U-Net with no backbone, U-Net DenseNet201, and CT-Unet EfficienNetV2S with a weight of 0.5, 0.3, and 0.2.

Test time augmentation was used for both tasks.

These models can be found here: https://drive.google.com/drive/folders/1FsF-B6xUvm2ZP7gcQ5ke7vl7XHlGSCsj?usp=sharing

## Task Description:
Buildings are essential to information regarding population, policy-making, and city management. Using computer vision technologies such as classification, object detection, and segmentation has proved helpful in several scenarios, such as urban planning and disaster recovery. Segmentation is the most precise method and can give detailed insights into the data as it highlights the area of interest.

Acquiring accurate segmentation masks of buildings is challenging since the training data derives from real-world photographs. As a result, the data often have varying quality, large class imbalance, and contain noise in different forms. The segmentation masks are affected by optical issues such as shadows, reflections, and perspectives. Additionally, trees, powerlines, or even other buildings may obstruct visibility. Furthermore, small buildings have proved to be more challenging to segment than larger ones as they are harder to detect, more prone to being obstructed, and often confused with other classes. Lastly, different buildings are found in several diverse areas, ranging from rural to urban locations. The diversity poses a vital requirement for the model to generalize to the various combinations.

**Task 1: Aerial Image Segmentation Task**

The aerial image segmentation task aims to solve the segmentation of buildings only using aerial images. Segmentation using only aerial images is helpful for several scenarios, including disaster recovery in remote sensing images where laser data is unavailable. We ask the participants to develop machine learning models for generating accurate segmentation masks of buildings solely using aerial images.

**Task 2: Laser Data Segmentation Task**

The laser data segmentation task aims to solve the segmentation of buildings using laser data. Segmentation using laser data is helpful for urban planning or change detection scenarios, where precision is essential. We ask the participants to develop machine learning models for generating accurate segmentation masks of buildings using laser data with or without aerial images.

## Code Description:
This repository's code is split between Jupyter Notebook files and Python files. The Jupyter Notebook files are mainly used for testing different concepts and visualizing. 

The scripts for this repository are in the code folder.

**Load_Dataset.py:** Downloads the MapAI dataset and creates a dataset folder where the dataset is stored.

**Preprocess_Dataset.py:** Preprocesses the MapAI dataset and creates subfolders containing the preprocessed data in the dataset folder.

**Train_Model.py:** Train a selected model on a chosen task. 

**Test_Model.py:** Tests and prints a model's IoU, BIoU, and Score. Users can choose between the validation and test sets. Users can also choose to enable TTA predictions.

**Model_Ensemble.py:** Tests and prints the IoU, BIoU, and Score of an ensemble of models. Users can choose between the validation and test sets. Users can also choose to enable TTA predictions. This script can run three types of ensemble methods.
1. 3 model ensemble with set weights.
2. Find the best weights for the ensemble on that specific set using three models. 
3. Find the best ensemble and weights on that specific set using a list of models.

**UNet.py:** Contains the code for the U-Net architecture.

**CTUNet.py:** Contains the code for the CT-Unet architecture.

**Loss_Metrics.py:** Contains the code for IoU and Dice coefficient metric and loss function.

**eval_functions.py:** Contains the code of the evaluation functions used for the MapAI competition. This code is taken from https://github.com/Sjyhne/MapAI-Competition/blob/master/competition_toolkit/competition_toolkit/eval_functions.py

**utils.py:** Contains the code for loading the dataset for testing and test time augmentation.



