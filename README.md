# Master Thesis: A Machine Learning approach for Building Segmentation using laser data

## Results:
|   Team   | IoU Task 1 | BIoU Task 1 | Total Task 1 | IoU Task 2 | BIoU Task 2 | Total Task 2 |    Score   | 
|:--------:|:------------:|:-------------:|:--------------:|:------------:|:-------------:|:--------------:|:----------:|
| Me |    0.7958    |     0.6226    |     0.7092     |    0.8918    |     0.7929    |     0.8423     | **0.7758** |
| 1st MapAI competition |    0.7794    |     0.6115    |     0.6955     |    0.8775    |     0.7857    |     0.8316     | **0.7635** |

**Task 1**
Results were achieved by using model ensembles on CT-Unet EfficientNetB4(Building masks), CT-Unet DenseNet201(Building masks) and CT-Unet DenseNet201(Edge masks) with the weight of 0.3, 0.3, 0.4.

**Task 2**
Results were achieved by using model ensembles on CT-Unet EfficientNetB4(Building masks), CT-Unet DenseNet201(Building masks) and CT-Unet ResNet50V2(Building masks) with the weight of 0.5, 0.2, 0.3.

## Task Description:
Buildings are essential to information regarding population, policy-making, and city management. Using computer vision technologies such as classification, object detection, and segmentation has proved helpful in several scenarios, such as urban planning and disaster recovery. Segmentation is the most precise method and can give detailed insights into the data as it highlights the area of interest.

Acquiring accurate segmentation masks of buildings is challenging since the training data derives from real-world photographs. As a result, the data often have varying quality, large class imbalance, and contains noise in different forms. The segmentation masks are affected by optical issues such as shadows, reflections, and perspectives. Additionally, trees, powerlines, or even other buildings may obstruct visibility. Furthermore, small buildings have proved to be more difficult to segment than larger ones as they are harder to detect, more prone to being obstructed, and often confused with other classes. Lastly, different buildings are found in several diverse areas, ranging from rural to urban locations. The diversity poses a vital requirement for the model to generalize to the various combinations.

**Task 1: Aerial Image Segmentation Task**

The aerial image segmentation task aims to solve the segmentation of buildings only using aerial images. Segmentation using only aerial images is helpful for several scenarios, including disaster recovery in remote sensing images where laser data is unavailable. We ask the participants to develop machine learning models for generating accurate segmentation masks of buildings solely using aerial images.

**Task 2: Laser Data Segmentation Task**

The laser data segmentation task aims to solve the segmentation of buildings using laser data. Segmentation using laser data is helpful for urban planning or change detection scenarios, where precision is essential. We ask the participants to develop machine learning models for generating accurate segmentation masks of buildings using laser data with or without aerial images.

## Code Description:
The code in this repository is split up between Jupyter Notebook files and Python files. The Jupyter Notebook files are mainly used for testing different concepts and visualizing the results before exporting the code to Python files. 

When running the code the python scripts in the code folder should be run in numerical order.
**0_Load_Dataset.py: **
Load
**1_Preprocess_Dataset.py: **
**2_Train_Model.py: **
**3_Test_Model.py: **
**4_Model_Ensemble.py: **
**CTUNet.py: **
**Loss_Metrics.py: **
**UNet.py: **
**eval_functions.py: **
**utils.py: **



