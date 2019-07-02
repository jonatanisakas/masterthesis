import numpy as np
import time

#Import necessary Sklearn library
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

#Import necessary Statsmodel libraries
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.bandwidths import bw_scott, bw_silverman



class KernelRegressionModel():
    
    def __init__(self, df, bandwidth='bw_scott'):
        
        self.df = df
        self.bandwidth = bandwidth
        self.efficient = False
        
    def model(self):
        
        #Time the modelling
        start_time = time.clock()
        
        #Extract dependent and independent variables
        y = self.df['impl_volatility'].values
        x = self.df[['strike_price', 'stock', 'T','riskfree']].values
        
        #Activate efficient bandwidth selection
        if self.bandwidth == None:
            self.efficient = True
            self.bandwidth = 'cv_ls'
            print('No predetermined bandwidth selected. Looking for optimizng the bandwidth')
            
        #Bandwidth defined by Scott D.W.
        elif self.bandwidth == 'bw_scott':
            self.bandwidth = bw_scott(x)
            #self.bandwidth = self.bandwidth*()
            print ('Selected bandwidth: ', self.bandwidth)
            
        #SBandwidth defined by Silverman B.W.
        elif self.bandwidth == 'bw_silverman':
            self.bandwidth = bw_silverman(x)
            print ('Selected bandwidth: ', self.bandwidth)
        
        #Or else select own bandsidth for the array
        else:
            pass
        
        #Optimize the bandwidth selection if no other bandwidth selection method is defined. 
        #See more here on their github page
        #https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/_kernel_base.py
        defaults = EstimatorSettings(efficient = self.efficient , randomize = False, n_sub = 50, n_res = 50, n_jobs = 0, return_only_bw = True )
        
        #Preprocess the data for faster computation
        x = preprocessing.normalize(x)
        
        #Split the data into traning anf testing data for in and out of sample testing
        xtrain, xtest, ytrain, ytest = train_test_split(x,y)
        
        #Define the regressor, with conrinues variables and the bandwith selection
        reg = KernelReg(endog = ytrain, exog = xtrain, var_type='cccc',  bw=self.bandwidth, defaults=defaults)
        
        #Fit the data onto the test data to get a out of sample prediction
        pred = reg.fit(xtest)[0]
        
        #Get the results from the test i form om RMSE and in and out of sample R^2 
        print('RMSE: ', np.sqrt(mean_squared_error(ytest, pred)))
        print('Out of Sample  R^2 :' , r2_score(ytest, pred))
        #print ('In sample ' , reg.r_squared())
        
        #Print the computing time
        print ('Estimation time: ', time.clock() - start_time, "seconds")
        
        return reg
        
        
        
        
        
        