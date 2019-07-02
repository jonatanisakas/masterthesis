from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



#start_time = time.clock()

#df = pd.read_csv('M:/Master thesis/Data/Results/final/call_option_fixed_vol.csv', sep=',', error_bad_lines=False,  dtype={'impl_volatility': float, 'moneyness':float,  'strike': int, 'stock': float,  'T': float,'riskfree':float} )

df = pd.read_csv('M:/Master thesis/Data/Results/final/call_option_fixed_vol.csv', sep=',', error_bad_lines=False,  dtype={'impl_volatility': float, 'moneyness':float,  'strike': int, 'stock': float,  'T': float,'riskfree':float} )

df = df.dropna()

y = df['impl_volatility'].values
x = df[['strike_price', 'stock', 'T','riskfree']].values



x = preprocessing.normalize(x)


xtrain, xtest, ytrain, ytest = train_test_split(x,y)

def build_model():
    model = keras.Sequential([
    layers.Dense(100, activation=tf.nn.relu, input_shape=[4]),
    layers.Dense(1)
      ])
    
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()
print(model.summary())

example_result = model.predict(xtest)


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 600


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(xtrain, ytrain, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])



hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,0.08])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,0.01])
  plt.legend()
  plt.show()


plot_history(history)


test_predictions = model.predict(xtest).flatten()
print(ytest)
print(test_predictions)

plt.scatter(ytest, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])





