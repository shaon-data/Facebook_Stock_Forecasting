import quandl, math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

df = quandl.get('WIKI/FB')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'

#instead of Nan, considering as outlier for not loosing nan row other information
#inplace used for instead of reassining a slice of previous dataframe into result varriable, we directly changed the dataframe
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.02*len(df))) #getting last 30 days forecasting

#Shift index by desired number of periods with an optional time freq
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train) #train
##saving the classifier for avoiding training every time after training once
with open('linearregression.pickle','wb') as f: #writing in binary format
        pickle.dump(clf, f) #serializationn of any python object and saving

pickle_in = open('linearregression.pickle','rb') # reading of binary format file
clf = pickle.load(pickle_in) # loading
        
accuracy = clf.score(X_test, y_test) #test

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.title('Forecasting of Facebook')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('pics/result.png')
plt.show()

