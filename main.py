import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import seaborn as sns
sns.set()

yf.pdr_override()
# Insert desired ticker
ticker = 'TSLA'
# Set number of days to predict
forecast_out = 5
df_full = pdr.get_data_yahoo(ticker, start="2010-01-01").reset_index()
df_full.to_csv('output/' + ticker + '.csv',index=False)

# reading df_full in case we want to read previously fetched data, can comment out previous 2 lines
df_full = pd.read_csv('output/' + ticker + '.csv')
df = df_full.copy()
df = df[['Adj Close']]


df['Prediction'] = df[['Adj Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
y = np.array(df['Prediction'])
y = y[:-forecast_out]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Support Vector Machine Regression:
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
svm_confidence = "{:.2%}".format(svm_confidence)
print("SVM confidence: ", svm_confidence)

# Linear Regression  Model:
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
lr_confidence = "{:.2%}".format(lr_confidence)
print("LR confidence: ", lr_confidence)

# Kernel Ridge Regression:
kridge = KernelRidge(alpha=1.0)
kridge.fit(x_train, y_train)
kridge_confidence = kridge.score(x_test, y_test)
kridge_confidence = "{:.2%}".format(kridge_confidence)
print("Kernel Ridge Confidence: ", kridge_confidence)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
lr_prediction = lr.predict(x_forecast)
svm_prediction = svr_rbf.predict(x_forecast)
kridge_prediction = kridge.predict(x_forecast)



last_date = datetime.datetime.strptime(df_full['Date'].iloc[-1], '%Y-%m-%d')
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)
df_full['LR Prediction'] = np.NaN
df_full['SVM Prediction'] = np.NaN
df_full['Ridge Prediction'] = np.NaN


for i, j, k in zip(lr_prediction, svm_prediction, kridge_prediction):
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    next_date_str = next_date.strftime('%Y-%m-%d')
    df_full.loc[len(df_full)] = [next_date_str,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,i,j,k]

print(df_full.tail(10))
df_full.tail(100).plot(kind='line',x='Date',y=['Adj Close','LR Prediction','SVM Prediction','Ridge Prediction'])
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title(ticker)
plt.savefig('output/'+ticker+'.png')
plt.show()