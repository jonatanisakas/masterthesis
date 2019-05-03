import numpy as np

#Set random seed

class HestonModel:
    
    '''
    In stochastic volatility models, the asset price and its volatility are both assumed to be random processes and can change over time. 
    There are many stochastic volatility models. Here we will present the most well-known and popular one: 
    the Heston Model. In Heston model, the stock price is log-normal distributed, the volatility process 
    is a positive increasing function of a mean-reversion process. 
    
    https://www.quantconnect.com/tutorials/introduction-to-options/local-volatility-and-stochastic-volatility
    '''
    
    def __init__(self, S0, K, sigma, T, rf, iterations, periods, longvol, gamma, voldrift, corr):
        
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.longvol = longvol #Long run Annualized volatility
        self.gamma = gamma #Volatility of volatility
        self.voldrift = voldrift #% speed of reversion to long run volatility
        self.corr = corr # Correlation between w1 and w2
        
    
    def optionPricingStochasticVolatility(self):
        
        option_data = np.zeros([self.iterations, 2])
        delta_t = self.T/(self.periods)
            
        stockPrice = np.zeros([self.iterations, self.periods + 1])
        variance = np.zeros([self.iterations, self.periods + 1])
        stockPrice[:, 0] = self.S0
        variance[:, 0] = np.power(self.sigma, 2)
        
        w1 = np.random.normal(0, 1, [self.iterations, self.periods])
        w2 = self.corr * w1 + np.sqrt(1-np.power(self.corr,2)) * np.random.normal(0, 1, [self.iterations, self.periods])
            
        for i in range(self.periods):
            
            #Run the variance function
            variance[:, i+1] = self.volatiltyCalculations(w1, delta_t, i, variance)
            
            #Calculate the Stockprice
            stockPrice[:, i+1] = stockPrice[:, i] * np.exp(delta_t * (self.rf - 0.5 * variance[:, i]) + np.sqrt(variance[:, i] * delta_t) * w2[:, i])
            
        finalPrice = stockPrice[:, -1]
        
        option_data[:,1] = finalPrice - self.K
    
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
            
        return np.exp(-1.0*self.rf*self.T)*average
        
    #Stochastic Volatility Calculations
    def volatiltyCalculations(self, w1, delta_t, i, variance):  
        
        variance[:, i+1] = np.power((np.sqrt(variance[:, i]) + 0.5 * self.gamma * np.sqrt(delta_t) * w1[:, i]), 2) \
        - self.voldrift *(variance[:,i] - np.power(self.longvol, 2)) \
        * delta_t - 0.25 * np.power(self.gamma, 2) * delta_t
        return variance[:, i+1]
        
        
if __name__ == "__main__":
    S0 = 100
    K = 100
    sigma = 0.2
    T = 1
    rf=0.01
    iterations = 1000
    periods = 300
    longvol = 0.20
    gamma = 0.08
    voldrift = 0.02
    model = HestonModel(S0, K, sigma, T, rf, iterations, periods, longvol, gamma, voldrift)
    print(model.optionPricingStochasticVolatility)