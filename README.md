# INFO4150-Engineering_Informatics
Engineering Informatics

Below is a guide for navigating projects and source files. 
Each Folder is an assignment completed in class with MP's (Mini Projects) being larger in scale.


HW2: Anomaly Detection
  - Source file is .ipynb (Jupyter nb)
  - Goal: correctly identify only the two anomalies within test set (red dots) while reducing the amount of misclassified train data.
  - Process: Compare multivariate distribution models when trained on independent versus dependent features. Use contour plots for visualization.

MP1: Web-app for Sensor value prediction
  - Source file is "MP1_flaskapp.py" 
  - Goal: build web app that will predict Sensor 2 values based on user input for Sensor 1 value
    - Route 1: User inputs a Sensor 1 value and receives a Sensor 2 value back. Sensor 1 values have to be saved to db file on each user input
    - Route 2: Admin route should allow admin to train new predicitive model on new database which includes the inputted Sensor 1 values
    - Route 3: Display RMSE of model

MP2: Neural Network for Image Classification
  - Source file is .ipynb (Jupyter nb)
  - Goal: build CNN for image classification of CIFAR dataset
    - Write report for architecture of CNN including layers and forward propogation
  
  ** cifar_test.csv and cifar10.csv are not in this repository due to size.

MP3: Long Short Term Memory Model for Stock Prediction
  - Source file is .ipynb (Jupyter nb)
  - Instructor already provided the working model
  - Goal: Write report for model-building process including reasons for normalization, architecture, and training/testing loop
    - Report file is Word doc
