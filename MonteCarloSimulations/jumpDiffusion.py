import numpy as np

class JumpDiffusionModel():
    
    '''
    Monte Carlo simulation [1] of Merton's Jump Diffusion Model [2].
    The model is specified through the stochastic differential equation (SDE):
    dS(t)
    ----- = mu*dt + sigma*dW(t) + dJ(t)
    S(t-)
    '''
    
    def __init__(self, S0, K, sigma, T, rf, iterations, periods, lambda_j, mu_j, sigma_j, a, b):
        
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.rf = rf
        self.iterations = iterations
        self.periods = periods
        self.lambda_j = lambda_j
        self.mu_j = mu_j #Drift
        self.sigma_j = sigma_j
        self.a = a
        self.b = b        
        
        
    def optionPricingJumpDiffusion(self):
        
        delta_t = self.T/(self.periods)
        option_data = np.zeros([self.iterations, 2])
        
        #mean_Y = np.exp(self.a + 0.5*(np.power(self.b, 2)))
        #variance_Y = np.exp(2*self.a + np.power(self.b, 2) * (np.exp(np.power(self.b, 2) - 1)))
        
        #M drift and V Volatility
        
        #M = self.S0 * np.exp(self.mu_j*self.T + self.lambda_j*self.T*(mean_Y-1))
        #V = np.power(self.S0, 2) * (np.exp((2*self.mu_j + np.power(self.sigma, 2))*self.T + self.lambda_j * self.T (variance_Y + np.power(mean_Y, 2) - 1)) - np.exp(2*self.mu_j * self.T + 2 * self.lambda_j * self.T * (mean_Y - 1)))
        stockPrice = np.zeros([self.iterations, self.periods + 1])
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
        
        for i in range(self.periods): 
            
            stockPrice[:,i+1] = stockPrice[:, i] * np.exp((self.mu_j - np.power(self.sigma, 2)/2)*delta_t \
                      + self.sigma* np.sqrt(delta_t) * w1[:, i] + self.a * Poisson[:,i] + np.sqrt(self.b**2) \
                      * np.sqrt(Poisson[:, i]) * w2[:,i])
        
        
        finalPrice = stockPrice[:, -1]
        option_data[:,1] = finalPrice - self.K
        average = np.sum(np.amax(option_data, axis=1))/float(self.iterations)
        
            
        return np.exp(-1.0*self.rf*self.T)*average
            
        