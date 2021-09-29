# Data Science Portfolio - Divit Saini

I am a fellow Chemical Engineer who is currently pursuing **MSc Advanced Chemical Engineering** at **Imperial College London**. The inquisitiveness regarding Data-Driven approaches in the domain of engineering motivated me to explore the field of Data Science by completing a Post Graduate Diploma in Data Science and AI from the College of Engineering Pune, India. This portfolio entails completed Data Science end-to-end projects as well as different case studies specific to Machine Learning Algorithms. The projects are showcased in the form of jupyter notebooks implemented in Python as well as R. This is updated on a regular basis.

- **Email**: [sainidivit@gmail.com](sainidivit@gmail.com)
- **LinkedIn**: [linkedin.com/divitsaini](https://www.linkedin.com/in/divitsaini/)


Every **End-to-End Projects** includes the following methodology:
1. *Introduction*: Brief introduction to the project highlingting the problem statement along with the objective of the project.

2. *Dataset Description*: A description regarding the source of the dataset along with the variables used. This also highlights the performance metrics used for comparison
4. *Importing Dataset and libraries required*
5. *Exploratory Data Analysis (EDA)*: Gaining insights from the dataset by exploring every dependent and independent variables. A few tasks includes checking variables for missing values; finding the range of different variables before implementing standardisation; finding correlation between the variables etc. All of this is done using pandas library in Python and dplyr in R.
6. *Data Preprocessing*: Includes tasks like imputing missing values; Standardisation/Normalisation of the numeric dependent variables
7. *Feature Engineering*: Includes tasks like One-hot encoding; Label encoding; Feature extraction techniques such as Recursive feature elimination, Backward Elimination, random forest feature importance graph.
8. *Train/Test split*: Creationg feature matrix and target array. Then Splitting them into X_train, Y_train, y_test, y_test into a certain split ratio.
9. *Modeling*: First, Machine Learning Algorithms or Deep Neural Networks is used to train different models using train dataset. Then, we test the models on the test dataset and compare their performance metrics.
10. *Saving the model*: Creating a .pkl file to be used later
11. *Conclusion*: After analysing performance metrics for different algorithms, the chosen model with best results is mentioned here with the key highlights. It also includes the future scope of the project.
12. *Deployment of the model*: This is an undergoing step and will be updated shortly.

## End-to-End Machine Learning Projects

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Healthcare%20Analytics%20-%20COVID19/Stats_picture.png"> **[Healthcare Analytics - Covid 19](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Healthcare%20Analytics%20-%20COVID19/Healthcare_COVID19.ipynb)**

To accurately predict the Length of Stay for each patient on case by case basis so that the Hospitals can use this information for optimal resource allocation and better functioning. The length of stay is divided into 11 different classes ranging from 0-10 days to more than 100 days. 

**Language**: *Python*; **Frameworks**: *NumPy, Pandas, Seaborn, Matplotlib, scikit-learn*;
**Classifiers**: Xgboost, Random forest

#

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/R%20Project%20-%20Life%20Expectancy%20Prediction/What-is-the-life-expectancy-for-someone-with-dementia.png"> **[Life Expectancy Prediction - R](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/R%20Project%20-%20Life%20Expectancy%20Prediction/Life_Expectancy_R.ipynb)**

The aim of this report is to carry out predictive modelling of Life Expectancy of a patient using the GHO data. It is a Regression problem with the independent variable being a ratio variable. Hypothesis Testing was carried out to compare two variables in the EDA section.

**Language**: *R*; **Frameworks**: *dplyr, tidyr, ggplot2, caret*;
**Regressors**: Xgboost, Random Forest

#

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Quality%20of%20Health%20-%20Classification/GoodCare_PoorCare.jpg"> **[Healthcare Analytics - Good care/Poor care](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Quality%20of%20Health%20-%20Classification/Logistic_Regression_HealthCare.ipynb)**

The aim of this report is to use the medical history/reports of the patients given in the dataset to predict whether Good/Poor Care was given. It is a binary classification problem. Different feature extraction techniques such as forward/backward elimination and Recursive Feature Elimination are explored in depth.

**Language**: *Python*; **Frameworks**: *NumPy, Pandas, Seaborn, Matplotlib, scikit-learn*;
**Regressors**: Logistic Regression

#

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Credit%20Risk%20Modeling/R.jpg"> **[Credit Risk Modeling](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Credit%20Risk%20Modeling/CRMEndtoEnd.ipynb)**

The aim of this report is to to predict whether a loan applicant will fully repay or default on a loan. Then I build a machine learning model that returns the unique loan ID and a loan status label that indicates whether the loan will be fully paid or charged off. It is a binary classification problem.

**Language**: *Python*; **Frameworks**: *NumPy, Pandas, Seaborn, Matplotlib, scikit-learn*;
**Classifiers**: Xgboost, Random Forest, CART

#

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Price%20Prediction%20-%20Bags/baggage_dimensions_en_2.png"> **[Price Prediction Model - Bags](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Price%20Prediction%20-%20Bags/BAGS_ABC_by_Divit-Stats_model-FINAL.ipynb)**

The aim of this report is to to predict the cost to be set for a new variant of the kinds of bags based on the specified attributes using multiple linear regression model

**Language**: *Python*; **Frameworks**: *NumPy, Pandas, Seaborn, Matplotlib, scikit-learn*;
**Regressor**: Linear Regression

#

## End-to-End Deep Learning Projects

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/Deep%20Learning/Computer%20Vision/Image%20Classification/Waste_Classification_CNN/OIP.jpg"> **[Waste Classification - CNN](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/Deep%20Learning/Computer%20Vision/Image%20Classification/Waste_Classification_CNN/Waste_Classification.ipynb)**

In this project, we adopt a deep convolution neural network (CNN) approach for classifying waste images (Recycle and Organic). It highlights a step by step approach on how to implement a deep CNN to perform image classification problem on waste data (images). 

**Language**: *Python*; **Frameworks**: *NumPy, Tensorflow, keras*;
**Neural Networks**: Convolution Neural Networks

#

<img align="left" width="275" height="200" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/Deep%20Learning/Computer%20Vision/Image%20Classification/Malaria%20Cell%20Classification%20-%20Transfer_learning/Capture1.JPG"> **[Malaria Cell Classification - Transfer Learning](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/Deep%20Learning/Computer%20Vision/Image%20Classification/Malaria%20Cell%20Classification%20-%20Transfer_learning/Malaria_Cell_Classification_VGG16_RESNET.ipynb)**

In this project, I have used Convolutional Neural Network (CNN) using ResNet50 and VGG16 Pre-Trained Models to classify Malaria cell images as Parasitized or Uninfected taken from human blood samples. This notebook also mentions a performance comparison table between the two pre-trained models and the testing of images is carried out on the best performing model. The testing is carried out by creating a simple UI by gradio library. 

**Language**: *Python*; **Frameworks**: *NumPy, Tensorflow, keras*;
**Neural Networks**: Convolution Neural Networks, Transfer Learning (ResNet50 and VGG16)
