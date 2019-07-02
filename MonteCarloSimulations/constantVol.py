import numpy as np

class ConstantVolatilityModel():


    def __init__(self, S0, K, sigma, T, rf, iterations, periods, tick):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.tick = tick
        
    def callOptionPriceNormalDistribution(self):
        
        #Generate two 2 columns, first with zeros and the second collumn will store the payoff
        #First column of zeros: payoff function is max(0, St - K) for a call option
        #The first columns is always going to be zeros, we later replace the second collumn with the payoff, the amax function will then pick the higher value
        call_option_data = np.zeros([self.iterations, 2])
        put_option_data = np.zeros([self.iterations, 2])
        
        #Generate random numbers from a normal distribution, 1 dimensional array with as many items as iterations
        #One number for every iteration of the stock price
        rand = np.random.normal(0, 1, [1, self.iterations])
        
        #Equation for the stock price - Risk-neutral assumption
        #Risk-neutral assumption, hence the drift is equal to the rf. 
        stock_price = self.S0*np.exp(self.T*(self.rf - 0.5*np.power(self.sigma, 2))+self.sigma*np.sqrt(self.T)*rand)
        
        #Option value = max(S-E, 0): Payoff function
        call_option_data[:,1] = stock_price - self.K
        put_option_data[:, 1] = self.K - stock_price
        
        #Amax returns the max value of S-E and then we take the average for the montecarlo simulation
        average_call = np.sum(np.amax(call_option_data, axis=1))/float(self.iterations)
        average_put = np.sum(np.amax(put_option_data, axis=1))/float(self.iterations)

        
        #Discount the value back to t 
        call_price = np.exp(-1.0*self.rf*self.T)*average_call
        put_price = np.exp(-1.0*self.rf*self.T)*average_put
        
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
        
    