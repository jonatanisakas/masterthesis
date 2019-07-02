# TensorFlow and tf.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
import scipy.stats as si

import time
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import numpy as np
import sklearn.mixture as mix
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as stats
from numba import jit
import math
from multiprocessing import cpu_count

import scipy.optimize as optimize


start_time = time.clock()
"""
df = pd.read_csv('M:/Master thesis/Data/test2.csv', sep=',', error_bad_lines=False,  dtype={'impl_volatility': float, 'moneyness':float,  'strike': int, 'stock': float,  'T': float,'riskfree':float} )

df = df.dropna()

y = df['option_price'].values
K = df['strike_price'].values
S = df['stock'].values
T = df['T'].values
r = df['riskfree'].values
sj = np.ones(len(r))*0.05
vol = df['impl_volatility'].values
"""

#data
strike = 2580
T = 15/365
sigma = 0.093656
S = 2430.06
r = 0.012053606
mu = 0.012053606 *0.5
y = 47.95

"""
T = np.array([1]*n_sample)
S = df['stock'].loc[0]
#K_max, K_min = max(df['strike_price']), min(df['strike_price'])
K_max, K_min = S * 1.3, S * 0.7
K = np.linspace(K_min, K_max, n_sample)
delta = (K_max - K_min)/n_sample
S = np.array([S]*n_sample)
r = df['riskfree'].loc[0]
r = np.array([r]*n_sample)
"""
#x = df[['strike_price', 'stock', 'T','riskfree', 'impl_volatility']].values


"""
x = preprocessing.normalize(x)
xtrain, xtest, ytrain, ytest = train_test_split(x,y)
"""
"""
mixtures = 2

def gradienrDecent(y, theta, learning_rate=0.1, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros(iterations)
    for i in range(iterations):
        
        X = test(K)
        
        pred = X[0]*theta[0] + X[1] * theta[1]
        print(pred)
        theta = theta -(1/m)*learning_rate*(X.T.dot((pred - y)))
        theta_history[i,:] = theta.T
        cost_history[i] = calCost(theta, x, y)
        
        
def calCost(theta, x, y):
    
    m = len(y)
    
    pred = x.dot(theta)
    cost = (1/2*m) * np.sum(np.square(pred - y))
    return cost        

x_1 = []
x_2 = []

r = sj - r
def test(K):
    for n, K in enumerate(K):
        if K >= S[n]:
            d1 = (np.log(S[n] / K) + (r[n] + 0.5 * sigma[n] ** 2) * T[n]) / (sigma[n] * np.sqrt(T[n]))
            d2 = d1 - sigma[n]*np.sqrt(T[n])
            c_1 = S[n] * si.norm.cdf(d1, 0.0, 1.0) - (K * np.exp(-r[n] * T[n]) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            d1 = (np.log(S[n] / K) + (r[n] + 0.5 * sigma[n] ** 2) * T[n]) / (sigma[n] * np.sqrt(T[n]))
            d2 = d1 - sigma[n]*np.sqrt(T[n])
            c_2 = S[n] * si.norm.cdf(d1, 0.0, 1.0) - (K * np.exp(-r[n] * T[n]) * si.norm.cdf(d2, 0.0, 1.0))
    return c_1, c_2


theta = np.random.randn(2,1)
theta,cost_history,theta_history = gradienrDecent(y,theta)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
""" 
df = pd.read_csv('M:/Master thesis/Data/test2.csv', sep=',', error_bad_lines=False,  dtype={'impl_volatility': float, 'moneyness':float,  'strike': int, 'stock': float,  'T': float,'riskfree':float} )

df = df.dropna()

y = df['option_price'].values
K = df['strike_price'].values
S = df['stock'].values
T = df['T'].values
r = df['riskfree'].values

e = len(K)

"""
K = 2580
T = 15/365
S = 2430.06
r = 0.012053606
y = 47.95
"""

def pricing(params):
    
    sigma_1, sigma_2, sigma_3, mu_1, mu_2, mu_3, theta_1, theta_2, theta_3 = params
    list_1 = []
    list_2 = []
    list_3 = []
    for n, i in enumerate(K):
        d1 = (np.log(S / K) + ((mu_1 -r) + 0.5 * sigma_1 ** 2) * T) / (sigma_1 * np.sqrt(T))
        d2 = d1 - sigma_1*np.sqrt(T)
        x_1 = S * np.exp((mu_1 - r)*T) * si.norm.cdf(d1, 0.0, 1.0) - (K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        c_1 = theta_1 * x_1
        list_1.append(c_1)
    
        d1 = (np.log(S / K) + (mu_2 - r + 0.5 * sigma_2 ** 2) * T) / (sigma_2 * np.sqrt(T))
        d2 = d1 - sigma_2*np.sqrt(T)
        x_2 = S * np.exp((mu_2 - r)*T) * si.norm.cdf(d1, 0.0, 1.0) - (K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        c_2 = theta_2 * x_2
        list_2.append(c_2)
        
        d1 = (np.log(S / K) + (mu_3 - r + 0.5 * sigma_3 ** 2) * T) / (sigma_3 * np.sqrt(T))
        d2 = d1 - sigma_3*np.sqrt(T)
        x_3 = S * np.exp((mu_3 - r)*T) * si.norm.cdf(d1, 0.0, 1.0) - (K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        c_3 = theta_3 * x_3
        list_3.append(c_3)
        
    x_1, x_2, x_3 = np.array(list_1), np.array(list_2), np.array(list_3)
    
    #print((c_1 + c_2 + c_3) , y)
    #print(theta_1 + theta_2 + theta_3)
    return np.sqrt((y - (x_1 + x_2 + x_3))**2)

def objective(params):
    
    x = pricing(params)
    for i in range(e):
        print(x[i])
        return x[i]

        
def thetacon(x):
    return x[-3] + x[-2] + x[-1]

def constraint(x):
    return thetacon(x) - 1

cons = ({'type': 'eq', 'fun': constraint})


#Initial guesses for the parameters
sigma_1 = 0.035
sigma_2 = 0.010
sigma_3 = 0.20
mu_1 = 0.012053606 *1.5
mu_2 = 0.012053606 *1.1
mu_3 = 0.012053606 *0.8
theta_1 = 0.4
theta_2 = 0.6
theta_3 = 0.02

inital_guess = np.array([sigma_1, sigma_2, sigma_3, mu_1, mu_2, mu_3, theta_1, theta_2, theta_3])

result = optimize.minimize(objective, inital_guess, constraints=cons, method = 'SLSQP')

test = result.x
#test2 = thetacon(test)
#print(constraint(test))
#print(test2)

if result.success:
    fitted_params = result.x
    print(fitted_params)
    print(fitted_params[-3] + fitted_params[-2] + fitted_params[-1])
    
else:
    raise ValueError(result.message)




""" 

init_stable_prob = 0.5
init_volatile_prob = 0.5

# guesses at starting mean
init_stable_mean = 0.10
init_volatile_mean = -0.1

# guesses at starting std
init_stable_std = 0.10
init_volatile_std = 0.3

init_probs = np.array([init_stable_prob, init_volatile_prob])
init_means = np.array([init_stable_mean, init_volatile_mean])
init_sigmas = np.array([init_stable_std, init_volatile_std])

log_pdf = stats.norm.logpdf(data, loc=self.mu, scale=self.sigma)

wsum = np.sum(weights)

mu = np.sum(weights * data) / wsum
sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum) 

distributions = [Normal(init_means[0], init_sigmas[0]), 
                 Normal(init_means[1], init_sigmas[1])]
""" 