import numpy as np

class ConstantVolatilityModel():


    def __init__(self, S0, K, sigma, T, rf, iterations, periods):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        
    def callOptionPriceNormalDistribution(self):
        
        #Generate two 2 columns, first with zeros and the second collumn will store the payoff
        #First column of zeros: payoff function is max(0, St - K) for a call option
        #The first columns is always going to be zeros, we later replace the second collumn with the payoff, the amax function will then pick the higher value
        option_data = np.zeros([self.iterations, 2])
        
        #Generate random numbers from a normal distribution, 1 dimensional array with as many items as iterations
        #One number for every iteration of the stock price
        rand = np.random.normal(0, 1, [1, self.iterations])
        
        #Equation for the stock price - Risk-neutral assumption
        #Risk-neutral assumption, hence the drift is equal to the rf. 
        stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*np.power(self.sigma, 2))+self.sigma*np.sqrt(self.T)*rand)
        
        #Option value = max(S-E, 0): Payoff function
        option_data[:,1] = stock_price - self.K
        
        #Amax returns the max value of S-E and then we take the average for the montecarlo simulation
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
        
        #Discount the value back to t 
        return np.exp(-1.0*self.rf*self.T)*average
        