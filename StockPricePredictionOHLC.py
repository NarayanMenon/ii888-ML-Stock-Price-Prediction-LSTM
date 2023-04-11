# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
import preprocessing as pre

# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
df1 = pd.read_csv('SPX.csv')
print(df1.head())
print(df1.shape)

# PLOTTING ALL INDICATORS IN ONE PLOT
plt.plot(df1)
plt.show()

# # PREPARATION OF TIME SERIES DATASET
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
df1 = df1.reshape(-1,10)
print(df1)
print(df1.shape)


# TRAIN-TEST SPLIT
training_size=int(len(df1)*0.8)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]
print(train_data.shape,test_data.shape)



# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)

X_train, y_train = train_data[:,0:9], train_data[:,9]
X_test, ytest = test_data[:,0:9], test_data[:,9]

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# LSTM MODEL
model=Sequential()
model.add(LSTM(100,return_sequences=True,input_shape=(9,1)))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

# MODEL COMPILING AND TRAINING
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)

print(model)

# PREDICTION
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


print(type(ytest))
print(ytest.shape)


print(type(test_predict))
print(test_predict.shape)

#  ytest_df = pd.DataFrame(ytest)
# DE-NORMALIZING FOR PLOTTING
ytest=scaler.inverse_transform(ytest.reshape(-1,1))
print(ytest)

test_predict = scaler.inverse_transform(test_predict.reshape(-1,1))
print(test_predict)

# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(y_train,train_predict))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(ytest,test_predict))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS

# ytest_org=scaler.inverse_transform(ytest)
plt.plot(ytest)
plt.plot(test_predict)
plt.show()

print("PREDICTION FOR TOMORROW")
inputArr = [[4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4080],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4085],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4090],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4095],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4100],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4105],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4110],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4115],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4120],
            [4081.15,4107.32,4069.84,4105.02,4085.2,4109.5,4072.55,4109.1,4125]]
inputArr = np.reshape(inputArr,(10,9))
inputArr=scaler.fit_transform(inputArr.reshape(-1,1))
inputArr = np.reshape(inputArr,(10,9))
inputArr = inputArr.reshape(inputArr.shape[0],inputArr.shape[1] , 1)

tomorrow = model.predict(inputArr)
tomorrow = scaler.inverse_transform(tomorrow.reshape(-1,1))
print(tomorrow)





























































































