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

## End-to-End Projects

<img align="left" width="250" height="175" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Healthcare%20Analytics%20-%20COVID19/Stats_picture.png"> **[Healthcare Analytics - Covid 19](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/Healthcare%20Analytics%20-%20COVID19/Healthcare_COVID19.ipynb)**

To accurately predict the Length of Stay for each patient on case by case basis so that the Hospitals can use this information for optimal resource allocation and better functioning. The length of stay is divided into 11 different classes ranging from 0-10 days to more than 100 days. 

**Language**: *Python*; **Frameworks**: *NumPy, Pandas, Seaborn, Matplotlib, scikit-learn*;
**Classifiers**: Xgboost, Random forest

#

<img align="left" width="250" height="175" src="https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/R%20Project%20-%20Life%20Expectancy%20Prediction/What-is-the-life-expectancy-for-someone-with-dementia.png"> **[Life Expectancy Prediction - R](https://github.com/divitsaini/Data_Science_Portfolio/blob/main/End%20to%20End%20ML%20projects/R%20Project%20-%20Life%20Expectancy%20Prediction/Life_Expectancy_R.ipynb)**

The aim of this report is to carry out predictive modelling of Life Expectancy of a patient using the GHO data mentioned above on Jupyter Notebook IDE using R-programming language. It is a Regression problem with the independent variable being a ratio variables.

**Language**: *R*; **Frameworks**: *dplyr, tidyr, ggplot2, caret*;
**Classifiers**: Xgboost, Random forest


# Data Science Portfolio - Arch Desai
This Portfolio is a compilation of all the Data Science and Data Analysis projects I have done for academic, self-learning and hobby purposes. This portfolio also contains my Achievements, skills, and certificates. It is updated on the regular basis.


## Achievements
- Recipient of Outstanding Master of Engineering - Industrial Engineering Student Award.
- [Publication](https://phmpapers.org/index.php/phmconf/article/view/1292): Prognosis of Wind Turbine Gearbox Bearing Failures using SCADA and Modeled Data, Proceedings of the Annual Conference of the PHM Society 2020, Vol. 12 No. 1.
- Winner of a TAMU Datathon 2020 among 50+ teams.
- Recipient of TAMU Scholarship and Fee Waiver for excellent academic performance (4.0 GPA).

## Projects

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/telecom.jpg"> **[Customer Survival Analysis and Churn Prediction](https://github.com/archd3sai/Customer-Survival-Analysis-and-Churn-Prediction)**

In this project I have used survival analysis to study how the likelihood of the customer churn changes over time. I have also implementd a Random Forest model to predict the customer churn and deployed a model using flask webapp on Heroku. [App](https://churn-prediction-app.herokuapp.com/)  

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/instacart.jpeg"> **[Instacart Market Basket Analysis](https://github.com/archd3sai/Instacart-Market-Basket-Analysis)**

The objective of this project is to analyze the 3 million grocery orders from more than 200,000 Instacart users and predict which previously purchased item will be in user's next order. Customer segmentation and affinity analysis are also done to study user purchase patterns.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/1_cEaeMuTvINqIgyYQMSJWUA.jpeg"> **[Hybrid-filtering News Articles Recommendation Engine](https://github.com/archd3sai/News-Articles-Recommendation)**
 
A hybrid-filtering personalized news articles recommendation system which can suggest articles from popular news service providers based on reading history of twitter users who share similar interests (Collaborative filtering) and content similarity of the article and user’s tweets (Content-based filtering).

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/airplane.jpeg"> **[Predictive Maintenance of Aircraft Engine](https://github.com/archd3sai/Predictive-Maintenance-of-Aircraft-Engine)**

In this project I have used models such as RNN, LSTM, 1D-CNN to predict the engine failure 50 cycles ahead of its time, and calculated feature importance from them using sensitivity analysis and shap values. Exponential degradation and similarity-based models are also used to calculate its remaining life.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/960x0.jpg"> **[Wind Turbine Power Curve Estimation](https://github.com/archd3sai/Wind-Turbine-Power-Curve-Estimation)**

In this project, I have employed regression techniques to estimate the Power curve of an on-shore Wind turbine. Nonlinear trees based regression methods perform best as true power curve is nonlinear. XGBoost is implemented and optimized using GridSearchCV which yields lowest Test RMSE-6.404.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/phase1.jpg"> **[Multivariate Phase 1 Analysis](https://github.com/archd3sai/Multivariate-Phase-1-Analysis)** 

Objective of this project is to identify the in-control data points and eliminate out of control data points to set up distribution parameters for manufacturing process monitoring. I utilized PCA for dimension reduction and Hotelling T2 and m-CUSUM control charts to established mean and variance matrices.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/gdp.jpg"> **[What's the GDP of India?](https://github.com/archd3sai/Predicting-GDP-of-India)**

Objective of this project is to perform predictive assesment on the GDP of India through an inferential analysis of various socio-economic factors. Various models are compared and Stepwise Regression model is implemented which resulted in 5.7% Test MSE.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/loan.jpg"> **[Loan Default Prediction](https://github.com/archd3sai/Loan-Default-Prediction)** 

In this project I applied various classification models such as Logistic Regression, Random Forest and LightGBM to detect consumers who will default the loan. SMOTE is used to combat class imbalance and LightGBM is implemented that resulted into the highest accuracy 98.89% and 0.99 F1 Score.
