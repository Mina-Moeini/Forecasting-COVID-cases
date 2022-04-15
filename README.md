
# Forecasting COVID cases with ML/DL models

This worldwide pandemic impacts hundreds of thousands of people and causes thousands of deaths each day.
Predicting the number of new cases (you can do the same for new deaths too) during this period can be a useful step in predicting the costs and facilities required in the future. 
Data from this https://www.data.gouv.fr supplied by Santé Publique France (Public Health France).
This project aims to evaluate the performance and compare of linear and  multiple non-linear regression techniques and neural network architecture, such as linear regression, support-vector regression (SVR), Random Forest Regressor,LSTM,RNN,for COVID-19 new cases rate prediction .
The performance of reproduction rate prediction is measured mean squared error (MSE).


## Prerequisites
Time series forecasting problems should be re-framed as supervised learning problems.
The important concepts that it's better you know.

 - [Time series as Suppervised Learning](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
 - [Univariate / Multivariate](https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/)
 - [Single step / Multi step](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
 - [Stationary / Non Stationary](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/)
 - [Cross Validation in Time Series](https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/#h2_2)
 - [ACF Plot](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/)
 - [Lag Observation](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)
 - [Feature Selection](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
 - [Handle Missing Value](https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e)

 






	

## Description of Folders

 - [```Datacleaning```](https://github.com/Mina-Moeini/Forecasting-COVID-cases/blob/master/Datacleaning.ipynb)- Import raw dataset/Missing value/Correlation/Feature selection/Fill missing value with KNN
 - [```best-parameters```](https://github.com/Mina-Moeini/Forecasting-COVID-cases/blob/master/best_parameters.ipynb)- Find best parameters for ML models with ```RandomizedSearchCV()```
 - [```Forecasting_DL```](https://github.com/Mina-Moeini/Forecasting-COVID-cases/blob/master/Forecasting_DL.ipynb) - Import dataset/ checking for stationary / Split Train-Test / Convert to supervised / Scale Data / Define Models and fit / MSE / Predict next step / Result Comparision
 - [```Forecasting_ML```](https://github.com/Mina-Moeini/Forecasting-COVID-cases/blob/master/Forecasting_ML.ipynb) - Import dataset/ checking for stationary / Split Train-Test / Convert to supervised / Scale Data / Define Models and fit / MSE / Predict next step 
 - [```learning_curve```](https://github.com/Mina-Moeini/Forecasting-COVID-cases/blob/master/learning_curve.ipynb) - plot of train and test scores for each model
 
## Roadmap

- Additional browser support

- Add more integrations


## Documentation

 - [Learning Curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
 - [Long Short-Term Memory Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
 - [Recurrent Neural Networks](https://www.tensorflow.org/guide/keras/rnn)
 - [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
 - [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
 - [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)


## 🔗 Contact

|||
|-|-|
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:m.moeini67@gmail.com) |[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mina-moeini)

## Authors

 [@Mina-Moeini](https://github.com/Mina-Moeini)

