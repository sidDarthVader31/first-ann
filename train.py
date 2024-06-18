import numpy as np 
import pandas as pd
import tensorflow as tf

#read dataset 
dataset = pd.read_excel('Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

#split data set into training data and test data 
# this is important because we need some data to test
# against
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)

#initialize ann 
ann = tf.keras.models.Sequential()

#add input and first hidden layer , here units is equal to
#number of neurons
ann.add(tf.keras.layers.Dense(units =6, activation ='relu'))    

#add another hidden layer 
ann.add(tf.keras.layers.Dense(units =6, activation ='relu'))

#add output layer 
ann.add(tf.keras.layers.Dense(units =1))

#compile the ANN 
ann.compile(optimizer= 'adam', loss = 'mean_squared_error')

#train the ANN model 
ann.fit(x_train, y_train, epochs = 100)

#test the ann model against test data 
y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
