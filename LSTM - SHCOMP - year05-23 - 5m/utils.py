import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def getdata(filename):
    df = pd.read_csv(filename)
    df['Datetime']= pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['LnClose'] = np.log(df['Close'])
    return df

def plot_figure(data,data2=[], title='No Title', xlab = 'No label', ylab = 'No label'):
    #visualizing
    plt.figure(figsize=(12,8))
    plt.title(title)
    plt.plot(data,scalex=True)
    if len(data2)!=0:
        plt.plot(data2,scalex=True)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

def plot_scatter_xy(X, Y, title='No Title', xlab = 'No label', ylab = 'No label'):
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    fig.show()

def get_realized_volatility(df):
    #calculate the realized volatility in one day
    dct = defaultdict(list)
    date_lst = []
    realized_vol = []
    for i in df.index:
        dct[i.date()].append(df.loc[i]['LnClose'])
        if len(date_lst) == 0 or i.date() != date_lst[-1]:
            date_lst.append(i.date())

    for date in date_lst:
        lst = dct[date]
        sum_vol = 0
        for j in range(1,len(lst)):
            sum_vol += ((lst[j]-lst[j-1])*100)**2
        realized_vol.append(sum_vol)

    return realized_vol

def get_volume(df):  
    # calculate the total trading volume in one day
    dct = defaultdict(list)
    date_lst = []
    volume_lst = []

    for i in df.index:
        dct[i.date()].append(df.loc[i]['Volume'])
        if len(date_lst) == 0 or i.date() != date_lst[-1]:
            date_lst.append(i.date())

    for date in date_lst:
        volume_lst.append(sum(dct[date]))
    
    return volume_lst

def get_date(df):
    #just get date, nothing else
    date_lst = []
    for i in df.index:
        if len(date_lst) == 0 or i.date() != date_lst[-1]:
            date_lst.append(i.date())
    return date_lst

# [[[1], [2], [3], [4], [5]]] [6]
# [[[2], [3], [4], [5], [6]]] [7]
# [[[3], [4], [5], [6], [7]]] [8]

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

def split_to_training_val_test( X1 , y1 , training_len = 5, val_len = 1):
    X_train1, y_train1 = X1[:training_len], y1[:training_len]
    X_val1, y_val1 = X1[training_len:val_len], y1[training_len:val_len]
    X_test1, y_test1 = X1[val_len:], y1[val_len:]
    return X_train1, y_train1, X_val1, y_val1, X_test1, y_test1

def build_LSTM_model(WINDOWS_SIZE = 5):
    model1 = Sequential()
    model1.add(InputLayer((WINDOWS_SIZE, 1)))
    model1.add(LSTM(64))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1, 'linear'))
    return model1

def show_predict_result(model1, X, y):
    train_predictions = model1.predict(X).flatten()
    train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y})
    plot_figure(train_results['Train Predictions'],data2=train_results['Actuals'],title='Training Result',\
        xlab='Data',ylab='Realized Volatility')
    return train_results 

def RMSE( Predict_val, Actual_val ):
    return np.sqrt(np.mean((Actual_val-Predict_val)**2))

def MAE(Predict_val, Actual_val ):
    return np.mean(abs(Actual_val-Predict_val))

def MSE(Predict_val, Actual_val ):
    return np.mean((Actual_val-Predict_val)**2)

def R_square(Predict_val, Actual_val):
    s1 = sum((Actual_val-Predict_val)**2)
    s2 = sum((Actual_val-np.mean(Actual_val))**2)
    return 1-(s1/s2)

def RMSPE(Predict_val, Actual_val):
    t = (1 - Predict_val/Actual_val)**2
    return math.sqrt(np.mean(t))

def MSLE( Predict_val, Actual_val):
    s1 = np.log(1+Actual_val)
    s2 = np.log(1+Predict_val)
    return np.mean((s1-s2)**2)

def Predictions_Error(Predictions_val, Actual_val):
    print('MAE = ', MAE(Predictions_val, Actual_val))
    print('MSE = ', MSE(Predictions_val, Actual_val))
    print('RMSE = ', RMSE(Predictions_val, Actual_val))
    print('R_2 = ', R_square(Predictions_val, Actual_val))
    print('RMSPE = ', RMSPE(Predictions_val, Actual_val))
    print('MSLE = ', MSLE(Predictions_val, Actual_val))

def Moving_Average( X, Actual_val ):
    predictions_val = [np.mean(i) for i in X]
    plot_figure(predictions_val,Actual_val,title='Moving Average',\
        xlab='Data',ylab='Realized Volatility')
    #Predictions_Error(predictions_val,Actual_val)

def Normalized(train_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std

    return train_df