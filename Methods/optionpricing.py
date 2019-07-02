import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
import scipy.stats as si
import math
#import my kernel regression model
from kernelregression import KernelRegressionModel
from mixturemethods import MixtureModel


class OptionPricing():
    
    def __init__(self, data, n_samples, T, S, r, bandwidth, mixturemodels):
        
        self.data = data #A dataframe consisting of relevant variables
        self.n_samples = n_samples
        self.T = T
        self.S = S
        self.r = r
        self.bandwidth = bandwidth
        self.mixturemodels = mixturemodels
    
    def selectVariables(self):
        
        #Define the variables needed for pricing the option with BS-model
        self.T = np.array([self.T]*self.n_samples)
        self.S = np.array([self.S]*self.n_samples)
        self.r = np.array([self.r]*self.n_samples)
        
        #Make a even linspace for the strikes to interpolate the option prices
        K_max, K_min = self.S[0] * 1.3, self.S[0] * 0.7
        self.K = np.linspace(K_min, K_max, self.n_samples)
        
        #Define a delta
        #self.delta = (K_max - K_min)/self.n_sample
        #And forward moneyness
        self.moneyness = np.array(np.log(self.K/(self.S*np.exp(self.r))))
        
    
    """
    Kernel Regression
    """
                  
    def functionEstimationKernelRegression(self):
        
        #Inintzialise variables
        self.selectVariables()
        
        input_data = []
        
        #Create a list of the variables so the model can calculate the implied volatility based on the earlier esimated function
        for i in range(self.n_samples):
            a = self.K[i], self.S[i], self.T[i], self.r[i]
            input_data.append(a)
        
        #Preprocess the data as in the model
        preprocessed_data = preprocessing.normalize(np.array(input_data))
        
        #Call the model
        regressor = KernelRegressionModel(self.data, self.bandwidth).model()
        
        #Estimate the IV 
        sigma = regressor.fit(preprocessed_data)[0]
        
        #Plot the volatility
        fig, ax = plt.subplots()
        ax.plot(self.moneyness, sigma)
        
        ax.set(xlabel='Moneyness', ylabel='Volatility',
               title='Implied Volatility')
        ax.grid()
        plt.show()
        
        kernerl_regression_prices = self.optionPricingKernelRegression(sigma)
        
        density, K = self.riskNeutralDensity(kernerl_regression_prices)
        
        #Defnie delta
        delta = (max(K) - min(K))/len(K)
        
        #Print out the total probability area
        print ('Density Kernel Regression ',np.trapz(density, dx=delta))
        
        #Plot the risk-neutral density
        fig, ax = plt.subplots()
        ax.plot(K, density)
        
        ax.set(xlabel='Strike Price', ylabel='Risk-Neutral Probability',
               title='Risk-neutral Density - Kernel Regression Method')
        ax.grid()
        
        plt.show()
        
    def optionPricingKernelRegression(self, sigma):
        
        #retruns a numpy array of call option prices
        d1 = np.array((np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T)))
        d2 = np.array(d1 - sigma * np.sqrt(self.T))
        c = np.array(self.S * si.norm.cdf(d1, 0.0, 1.0) - (self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0)))
        return c
    
    """
    Mixture Model
    """
        
    def functionMixtureModel(self):
        
        self.selectVariables()
        
        mixture_model_parameters = MixtureModel(self.data, self.mixturemodels).model()
        
        density = self.mixtureOptionPricingModel(mixture_model_parameters)
        
        delta = (max(self.K) - min(self.K))/len(self.K)
        
        print ('Density Mixture Method ',np.trapz(density, dx=delta))

        fig, ax = plt.subplots()
        ax.plot(self.K, density)
        
        ax.set(xlabel='Strike Price', ylabel='Risk-Neutral Probability',
               title='Risk-neutral Density - Mixture Model')
        ax.grid()
        
        plt.show()
        
    
    def mixtureOptionPricingModel(self, parameters):
        
        density = 0
        
        l = self.mixturemodels
        sigma, mu, theta_list, weight = parameters[:l], parameters[l:2*l], parameters[2*l:3*l-1], parameters[-1]

        theta_1 = (np.sum(theta_list) - weight)*-1
        
        for i in range(self.mixturemodels):   
                
            if i == 0:
                theta = theta_1
            else:
                theta = theta_list[i-1]
            
            b = sigma[i] * np.sqrt(self.T)
            a = np.log(self.S) + (mu[i] - 0.5*(sigma[i]**2))*self.T
            
            L = theta * ((1/(self.K * b*np.sqrt(2*math.pi)))*np.exp(-(np.log(self.K)-a)**2/(2*(b**2))))
            
            density += L
    
        
        return density
    
    """
    Calculate the risk-neutral density
    """ 
    
    def riskNeutralDensity(self, option_prices):
        
        dc = np.diff(option_prices,1)
        dK = np.diff(self.K,1)
        
        #First derivative of C with respect to K
        c_first = dc/dK
        #Boundery check for the call prices
        c_first, K, r, T = self.boundaryCondition(c_first)
        
        
        K_first = 0.5*(self.K[:-1]+self.K[1:])
        #Second derivative of C with respect to K
        dc_first = np.diff(c_first, 1)
        dK_first = np.diff(K_first, 1)
        
        #Retrun the risk-neutral density
        density = (dc_first/dK_first)*np.exp(r[1:]*T[1:])
        
        #Define K
        K = 0.5*(K_first[:-1]+K_first[1:])
        
        return density, K
        
    #Boundery check for the call prices where dc/dK >= -e^-rt
    def boundaryCondition(self, c_first):
        
        dc, strikes, r, T = ([] for i in range(4))
        
        for i in c_first:
            if i > 0:
                print (i)
        
        for n, i in enumerate(c_first):
            if i >= -np.exp(self.r[n] * self.T[n]):
                dc.append(i)
                strikes.append(self.K[n])
                r.append(self.r[n])
                T.append(self.T[n])
            else:
                print (i ,': violated boundary condition')
        dc = np.asarray(dc)
        K = np.asarray(strikes)
        r = np.asarray(r)
        T = np.asarray(T)
        return dc, K, r, T
