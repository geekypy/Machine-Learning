#!python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#This python script is aimed to create a linear regression. Since our dataset has several
#feature, I will select one column only. Column NOGT
#Dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/Air+Quality#

#Attributes
path_file = "C:/Users/GeekyPy/AirQualityUCI.xlsx"

def generate_datasets(colnum):

    #Read the file, and process column #5 into a list for further processing
    df=pd.read_excel(path_file)
    columns = df.columns
    y_label = columns[colnum]
    #Put the excel file into a data-frame and select the NO2 column
    NO2_y = df['NO2(GT)']
    NO2_x = NO2_y.index
    #Generate training and validation sets
    #TRAINING 99 SAMPLES
    NO2_Train_X = NO2_x[1:100]
    NO2_Valid_X = NO2_x[101:200]
    #Validation 100 Samples
    NO2_Train_y = NO2_y[1:100]
    NO2_Valid_y = NO2_y[101:200]

    #Reshape Arrays
    NO2_Train_X = np.array(NO2_Train_X).reshape(-1,1)
    NO2_Train_y = np.array(NO2_Train_y).reshape(-1,1)
    NO2_Valid_X = np.array(NO2_Valid_X).reshape(-1,1)
    NO2_Valid_y = np.array(NO2_Valid_y).reshape(-1,1)

    return NO2_Train_X, NO2_Valid_X, NO2_Train_y, NO2_Valid_y, y_label

def Linear_Regression(NO2_Train_X, NO2_Valid_X, NO2_Train_y, NO2_Valid_y,y_label):

    #Instantiate the linear regression object
    lin_regr = linear_model.LinearRegression()
    #Train the Linear classifier.
    lin_regr.fit(NO2_Train_X,NO2_Train_y)
    #Generate the predictions--> For this we use the validation test
    no2_prediction = lin_regr.predict(NO2_Valid_X)
    #Plot the outputs and compare how far off we were
    plt.scatter(NO2_Valid_X,NO2_Valid_y, color='blue')
    plt.plot(NO2_Valid_X,no2_prediction, color = 'black')
    plt.xlabel("Samples Serial(s) ")
    plt.ylabel(y_label)
    plt.title("Linear Regression : "+y_label)
    #Regression coefficients
    coeff = lin_regr.coef_
    print("Linear Regression PARAMETERS")
    print("Coefficients: ", coeff)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(NO2_Valid_y, no2_prediction))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(NO2_Valid_y, no2_prediction))
    plt.show()

if __name__ == "__main__":

    colnum = 4
    generate_datasets(colnum)
    NO2_Train_X, NO2_Valid_X, NO2_Train_y, NO2_Valid_y, y_label = generate_datasets(4)
    Linear_Regression(NO2_Train_X, NO2_Valid_X, NO2_Train_y, NO2_Valid_y, y_label)