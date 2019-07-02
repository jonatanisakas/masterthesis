import numpy as np
import pandas as pd
from scipy.integrate import quad
import scipy
import scipy.stats as si
 
from math import sqrt, exp, log, pi
 
#Set random seed

class HestonModel:
    
    def __init__(self, S0, K, sigma, T, rf, iterations, periods, longvol, gamma, kappa, rho, tick):
        
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.v = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.longvol = longvol #Long run Annualized volatility
        self.gamma = gamma #Volatility of volatility
        self.kappa = kappa #% speed of reversion to long run volatility
        self.rho = rho # Correlation between w1 and w2
        self.tick = tick
        
        self.x = log(self.S0)
        lamb = 0
        
        #####
        self.u1 = 0.5
        self.u2 = -0.5
         
        self.b1 = self.kappa + lamb - self.rho*self.longvol
        self.b2 = self.kappa + lamb
        
        self.a = self.kappa*self.longvol
        
    
    def optionPricingStochasticVolatility(self):
        
        
        #w1 = np.random.normal(0, 1, [self.iterations, self.periods])
        #w2 = self.rho * w1 + np.sqrt(1-np.power(self.rho,2)) * np.random.normal(0, 1, [self.iterations, self.periods])
        delta_t = self.T/(self.periods)
        call_option_data = np.zeros([self.iterations, 2])
        put_option_data = np.zeros([self.iterations, 2])
        
        
        #Stock prices
        stockPrice = np.zeros([self.iterations, self.periods])
        stockPrice[:, 0] = self.S0
        #Volatility
        volatility = np.zeros([self.iterations, self.periods])
        volatility[:, 0] = self.sigma**2
        
        #Stochastic variance component
        w1 = np.random.normal(0, 1, [self.iterations, self.periods])
        w2 = self.rho * w1 + np.sqrt(1-np.power(self.rho,2)) * np.random.normal(0, 1, [self.iterations, self.periods])
            
        for i in range(1, self.periods):
            #Run the variance function
            
            
            
            #volatility[:, i] = np.power(np.sqrt(volatility[:, i-1]) + 0.5 * self.longvol * np.sqrt(delta_t) * w2[:, i],2) - self.kappa *  (np.maximum(0,volatility[:, i-1]) - self.longvol) * delta_t - 0.25*np.power(self.gamma,2) *delta_t
            #Milstein SCheme
            volatility[:, i] = volatility[:, i-1] + self.kappa * (self.longvol - np.maximum(0, volatility[:, i-1])) * delta_t + self.gamma*np.sqrt(np.maximum(0, volatility[:, i-1])*delta_t)*w2[:,i-1] + (0.25*self.gamma**2)*delta_t*(np.power(w2[:,i-1],2)-1)
            #volatility[:, i] = volatility[:, i-1] + self.kappa * (self.longvol - np.maximum(0, volatility[:, i-1]))*delta_t + self.gamma*np.sqrt(np.maximum(0, volatility[:, i-1])*delta_t)*w2[:, i]
            stockPrice[:, i] = stockPrice[:, i-1] * np.exp(((self.rf - 0.5 * np.maximum(0, volatility[:, i-1]))*delta_t  + np.sqrt(np.maximum(0, volatility[:, i-1])* delta_t) * w1[:, i-1]))

        finalPrice = stockPrice[:, -1]

        call_option_data[:,1] = finalPrice - self.K
        put_option_data[:, 1] = self.K - finalPrice
        
        average_call = np.sum(np.amax(call_option_data, axis=1))/float(self.iterations)
        average_put = np.sum(np.amax(put_option_data, axis=1))/float(self.iterations)

        call_price = np.exp(-1.0*self.rf*self.T)*average_call
        put_price = np.exp(-1.0*self.rf*self.T)*average_put
        
        #Add mispricing
        mispricing = np.random.uniform(-self.tick, self.tick)
        if call_price >= 3:
            call_price = call_price + mispricing*2
        if call_price >= 0.05:
            call_price = call_price + mispricing
        if put_price >= 3:
            put_price = put_price + mispricing*2
        if put_price >= 0.05:
            put_price = put_price + mispricing
        
        return call_price, put_price

        
if __name__ == "__main__":
    S0 = 100
    K = 105
    sigma = 0.166
    T = 0.4
    rf= 0.012986
    iterations = 10000
    periods = 200
    
    longvol = 0.10
    gamma = 0.5
    kappa = 3.5
    rho = -0.7
    tick = 0.025
    model = HestonModel(S0, K, sigma, T, rf, iterations, periods, longvol, gamma, kappa, rho, tick)
    model.optionPricingStochasticVolatility()