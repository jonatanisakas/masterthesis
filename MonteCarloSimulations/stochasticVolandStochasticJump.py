import numpy as np

"""
In mathematical finance, the stochastic volatility jump (SVJ) model is suggested by Bates.
This model fits the observed implied volatility surface well. The model is a Heston process with an added Merton log-normal jump.
"""

class StochasticVolandStochasticJumpModel():
    
    
    """Using the same parameters as in the jumpdiffusion and stochastic volatility models"""
    def __init__(self, S0, K, sigma, T, rf, iterations, periods, longvol, gamma, kappa, corr, lambda_j, mu_j, delta_j, tick):
        
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.longvol = longvol #Long run Annualized volatility
        self.gamma = gamma #Volatility of volatility
        self.kappa = kappa #% speed of reversion to long run volatility
        self.corr = corr # Correlation between w1 and w2
        self.lambda_j = lambda_j
        self.mu_j = mu_j #Drift
        self.delta_j = delta_j
        self.tick = tick
        
    def optionPricingStochasticVolandJump(self):
        
        delta_t = self.T/(self.periods)
        call_option_data = np.zeros([self.iterations, 2])
        put_option_data = np.zeros([self.iterations, 2])
        
        rj = self.lambda_j*(np.exp(self.mu_j + 0.5*self.delta_j**2)-1)
        
        #Stock prices
        stockPrice = np.zeros([self.iterations, self.periods])
        stockPrice[:, 0] = self.S0
        
        #Variance
        volatility = np.zeros([self.iterations, self.periods])
        volatility[:, 0] = np.power(self.sigma, 2)
        
        #Stochastic variance component
        v1 = np.random.normal(0, 1, [self.iterations, self.periods])
        v2 = self.corr * v1 + np.sqrt(1-np.power(self.corr,2)) * np.random.normal(0, 1, [self.iterations, self.periods])
        
        #Jump component
        #w1 = np.random.normal(0,1, [self.iterations, self.periods])
        w2 = np.random.normal(0,1, [self.iterations, self.periods])
        Poisson = np.random.poisson(self.lambda_j*delta_t,[self.iterations, self.periods])
        
        #Let's loop it
        for i in range(1, self.periods):
            
            #Run the variance function
            volatility[:, i] = volatility[:, i-1] + self.kappa * (self.longvol - np.maximum(0, volatility[:, i-1])) * delta_t + self.gamma*np.sqrt(np.maximum(0, volatility[:, i-1])*delta_t)*v2[:,i-1] + (0.25*self.gamma**2)*delta_t*(np.power(v2[:,i-1],2)-1)
            
            #volatility[:, i] = np.power(np.sqrt(volatility[:, i-1]) + 0.5 * self.longvol * np.sqrt(delta_t) * w2[:, i],2) - self.kappa *  (np.maximum(0,volatility[:, i-1]) - self.longvol) * delta_t - 0.25*np.power(self.gamma,2) *delta_t
            
            #volatility[:, i] = volatility[:, i-1] + self.kappa * (self.longvol - np.maximum(0, volatility[:, i-1]))*delta_t + self.gamma*np.sqrt(np.maximum(0, volatility[:, i-1])*delta_t)*w2[:, i]
            
            #Add jump diffusion and the stochastic volatility component variance and v2
            stockPrice[:,i] = stockPrice[:, i-1] * (np.exp((self.rf - rj - 0.5 * np.maximum(0, volatility[:, i-1])) * delta_t  + np.sqrt(np.maximum(0, volatility[:, i-1])) * np.sqrt(delta_t) * v1[:, i-1]) + (np.exp(self.mu_j + self.delta_j* w2[:, i-1])-1)*Poisson[:,i])
             
            
        finalPrice = stockPrice[:, -1]
        
        
        call_option_data[:,1] = finalPrice - self.K
        put_option_data[:, 1] = self.K - finalPrice
        
        average_call = np.sum(np.amax(call_option_data, axis=1))/float(self.iterations)
        average_put = np.sum(np.amax(put_option_data, axis=1))/float(self.iterations)
        
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
        
if __name__ == "__main__":
    
    S0 = 100
    K = 60
    sigma = 0.166
    T = 0.4
    rf= 0.012986
    iterations = 400000
    periods = 300

    
    #For Stochastic Volatility
    longvol = 0.10
    gamma = 0.5
    kappa = 3.5
    rho = -0.7
    
    #For JumpDiffusion
    lambda_j = 0.5 #Jump frequency
    mu_j = -0.03 #Expected jump size
    delta = 0.15 #Jump size volatility
    
    model = StochasticVolandStochasticJumpModel(S0, K, sigma, T, rf, iterations, periods, longvol, gamma, kappa, rho, lambda_j, mu_j, delta)
    model.optionPricingStochasticVolandJump()
                    