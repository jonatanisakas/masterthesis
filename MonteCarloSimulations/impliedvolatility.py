from math import log, sqrt, exp
from scipy import stats
from scipy.optimize import fsolve


class ImpliedVolatility():
    
    def __init__(self, S0, K, T, rf, sigma):
        
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.rf = rf
        self.sigma = sigma
    
    def d1(self):
        d1 = ((log(self.S0 / self.K) + (self.rf + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        return d1
    
    
    def value(self):
    
        d1 = self.d1()
        d2 = ((log(self.S0 / self.K) + (self.rf - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        value_call = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.rf * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        value_put = (self.K * exp(-self.rf * self.T) * stats.norm.cdf(-d2, 0.0, 1.0))- (self.S0 * stats.norm.cdf(-d1, 0.0, 1.0))
        return value_call, value_put
    
    def vega(self):
        vega = self.S0 * stats.norm.pdf(self.d1(), 0.0, 1.0) * sqrt(self.T)
        return vega
    
    def imp_vol(self, C0, P0, sigma_est=0.166):
        option = ImpliedVolatility(self.S0, self.K, self.T,self.rf, sigma_est)

        def difference_call(sigma):
            option.sigma = sigma
            value_call = option.value()[0]
            return value_call - C0
        def difference_put(sigma):
            option.sigma = sigma
            value_put = option.value()[1]
            return value_put - P0
        return fsolve(difference_call, sigma_est)[0], fsolve(difference_put, sigma_est)[0]