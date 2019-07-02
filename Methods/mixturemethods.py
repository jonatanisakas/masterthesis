from sklearn.metrics import r2_score
import scipy.stats as si
import scipy.optimize as optimize

# Helper libraries
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format

import random
import time

class MixtureModel():
    
    def __init__(self, df, mixturemodels, maxiter=40000):
        
        self.df = df
        self.mixturemodels = mixturemodels
        self.maxiter = maxiter
       
        start_time = time.clock()

    def model(self):
        
        #Initialize model variables
        self.S = self.df['stock'].values
        self.K = self.df['strike_price'].values
        self.r = self.df['riskfree'].values
        self.T = self.df['T'].values
        self.y = self.df['option_price'].values
        
        
        inital_guess, bound = self.gerenrateInitialGuessValues()
        inital_guess = np.asarray(inital_guess)
        
        result = optimize.least_squares(self.optionPricingModel, inital_guess, bounds = bound, max_nfev = self.maxiter, ftol = 1e-08)
        return result.x
        
        
        #Split the data so there is a vector of parameters for each of the time to maturity, thus one set of parameters Z for [t, T]
    def optionPricingModel(self, inital_guess):
        
        l = self.mixturemodels
        sigma, mu, theta_list, weight = inital_guess[:l], inital_guess[l:2*l], inital_guess[2*l:3*l-1], inital_guess[-1]
        #call_option_prices = np.zeros([len(self.K)])
        theta_1 = (np.sum(theta_list) - weight)*-1
        
        #print('theta 1', theta_1, 'theta 2', theta_list)
        
        call_payoff = 0
        
        for i in range(self.mixturemodels):
            if i == 0:
                theta = theta_1
            else:
                theta = theta_list[i-1]
            
            a = np.log(self.S) + (mu[i] - 0.5*(sigma[i]**2))*self.T
            b = sigma[i] * np.sqrt(self.T)
            
            d1 = (-np.log(self.K)+a+b**2)/b
            d2 = d1 - b
            
            mixture = theta * (np.exp(a+0.5*(b**2)) * si.norm.cdf(d1, 0.0, 1.0) - self.K * si.norm.cdf(d2, 0.0, 1.0))
            
            call_payoff += mixture

        call_option_prices = np.exp(-self.r*self.T)*call_payoff
        
        return self.y - call_option_prices
    
        
    def gerenrateInitialGuessValues(self):
        
        sigma, bounds_lower_sigma, bounds_higher_sigma, mu, bounds_lower_mu, bounds_higher_mu, \
        theta, bounds_lower_theta, bounds_higher_theta = (np.array([]) for i in range(9))
        
        #Total Theta weight that sums up to one
        weight = np.array([1])
        bounds_lower_weight, bounds_higher_weight = np.array([0.999]),  np.array([1])
        
        #Generate random variables
        for i in range(self.mixturemodels):
            
            #Generate random variables
            sigma = np.append(sigma, (random.uniform(0, 1)))  
            mu = np.append(mu, (random.uniform(-1, 1)))
            theta = np.append(theta, (random.uniform(0, 1)))
            
            #Create Bounds
            bounds_lower_sigma = np.append(bounds_lower_sigma, 0)
            bounds_higher_sigma = np.append(bounds_higher_sigma, np.inf)
            
            bounds_lower_mu = np.append(bounds_lower_mu, -np.inf)
            bounds_higher_mu = np.append(bounds_higher_mu, np.inf)
            
            bounds_lower_theta = np.append(bounds_lower_theta, 0)
            bounds_higher_theta = np.append(bounds_higher_theta, 1)
        
        #Exclude first theta since theta_1 = weight - sum(theta_i)
        bounds_lower_theta = bounds_lower_theta[1:]
        bounds_higher_theta = bounds_higher_theta[1:]
        
        theta = theta[1:]
        
        initial_guess = np.concatenate((sigma, mu, theta, weight))
        bounds_lower = np.concatenate((bounds_lower_sigma, bounds_lower_mu, bounds_lower_theta, bounds_lower_weight))
        bounds_higher = np.concatenate((bounds_higher_sigma, bounds_higher_mu, bounds_higher_theta, bounds_higher_weight))
        bounds = bounds_lower, bounds_higher
        
        return initial_guess, np.array(bounds)
            
if __name__ == "__main__":
    df = pd.read_csv('M:/Master thesis/Data/test.csv', sep=',', error_bad_lines=False,  dtype={'impl_volatility': float, 'moneyness':float,  'strike': int, 'stock': float,  'T': float,'riskfree':float} )
    print(MixtureModel(df, 3).model()) 
        