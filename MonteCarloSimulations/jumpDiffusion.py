import numpy as np

class JumpDiffusionModel():
    
    '''
    Monte Carlo simulation [1] of Merton's Jump Diffusion Model [2].
    The model is specified through the stochastic differential equation (SDE):
    dS(t)
    ----- = mu*dt + sigma*dW(t) + dJ(t)
    S(t-)
    '''
    
    def __init__(self, S0, K, sigma, T, rf, iterations, periods, lambda_j, mu_j, delta_j, tick):
        
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.lambda_j = lambda_j # jump frequency p.a.
        self.mu_j = mu_j # expected jump size
        self.delta_j = delta_j # jump size volatility
        self.tick = tick
        
    def optionPricingJumpDiffusion(self):
        
        delta_t = self.T/(self.periods)
        call_option_data = np.zeros([self.iterations, 2])
        put_option_data = np.zeros([self.iterations, 2])
        
        #Drift correction for jump
        rj = self.lambda_j*(np.exp(self.mu_j + 0.5*self.delta_j**2)-1) 
        
        #mean_Y = np.exp(self.a + 0.5*(np.power(self.b, 2)))
        #variance_Y = np.exp(2*self.a + np.power(self.b, 2) * (np.exp(np.power(self.b, 2) - 1)))
        
        #M drift and V Volatility
        
        #M = self.S0 * np.exp(self.mu_j*self.T + self.lambda_j*self.T*(mean_Y-1))
        #V = np.power(self.S0, 2) * (np.exp((2*self.mu_j + np.power(self.sigma, 2))*self.T + self.lambda_j * self.T (variance_Y + np.power(mean_Y, 2) - 1)) - np.exp(2*self.mu_j * self.T + 2 * self.lambda_j * self.T * (mean_Y - 1)))
        stockPrice = np.zeros([self.iterations, self.periods])
        #K = np.zeros([self.iterations, self.periods + 1]) 
        #K[:, 0] = self.K
        stockPrice[:, 0] = self.S0
    
        ''' 
        To account for the multiple sources of uncertainty in the jump diffusion process, generate three arrays of random variables.
        - The first one is related to the standard Brownian motion, the component epsilon(0,1) in epsilon(0,1) * np.sqrt(dt);
        - The second and third ones model the jump, a compound Poisson process:
        the former (a Poisson process with intensity Lambda) causes the asset price to jump randomly (random timing); the latter (a Gaussian variable) 
        defines both the direction (sign) and intensity (magnitude) of the jump.
        
        '''
        
        w1 = np.random.normal(0,1, [self.iterations, self.periods])
        w2 = np.random.normal(0,1, [self.iterations, self.periods])
        
        Poisson = np.random.poisson(self.lambda_j*delta_t,[self.iterations, self.periods])
        
        for i in range(1, self.periods): 
           
            stockPrice[:,i] = stockPrice[:, i-1] * (np.exp((self.rf - rj - 0.5*self.sigma**2)*delta_t + self.sigma*np.sqrt(delta_t) * w1[:,i-1]) + (np.exp(self.mu_j + self.delta_j* w2[:,i-1])-1)*Poisson[:,i-1])
            
            """
            stockPrice[:,i+1] = stockPrice[:, i] * np.exp((self.mu_j - np.power(self.sigma, 2)/2)*delta_t \
                      + self.sigma* np.sqrt(delta_t) * w1[:, i] + self.a * Poisson[:,i] + np.sqrt(self.b**2) \
                      * np.sqrt(Poisson[:, i]) * w2[:,i])
            """ 
            
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
    sigma = 0.25
    T = 1
    rf= 0.012986
    iterations = 100000
    periods = 200
    
    lambda_j = 0.5 #Jump frequency
    mu_j = -0.04 #Expected jump size
    delta = 0.20 #Jump size volatility
    tick = 0.025
    
    model = JumpDiffusionModel(S0, K, sigma, T, rf, iterations, periods, lambda_j, mu_j, delta, tick)
    model.optionPricingJumpDiffusion()