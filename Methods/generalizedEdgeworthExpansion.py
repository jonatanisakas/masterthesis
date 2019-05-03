


class GeneralizedEdgeworthExpansion():
    
    '''
    Generalized Edgeworth series expansion, has the
    desirable property that the coefficients in the expansion are simple function
    of the moments of the given and approximating distributions. 
    '''
    def __init__(self):
        pass
        
        
    def model(self):
        '''
        C(A) corresponds to the Black Scholes formula
        then the expansion will give an explicit expression for the adjustment terms between the rue option value, C(F)
        and Black Scholes formula C(A).
        
        The approcimating lognormal distribution for the stock price, St is a function of two parameters: the first and
        seocnd cumulants of the random variable log(St). 
        Alpha(A) = true distribution alpha(F) = S0 * exp(rf * T) - Risk neutrality 
        
        In the paper they use the second method: equate the second cumulatns of log(St) for the approcimating lognormal
        and the true distribution. 
        
        The distribution is given by 
        a(St) = sqrt(St * sigma * sqrt*(t * 2 * pi))* exp(-(log(St) - (log(alpha1(A))-variance*t/2))^2*variance*t)
        alpha1(A) = S0 * e^rf*t
        variance * t = Intergral(log(St)^2 dF(St)) - [Intergral(log(St)dF(St))]^2 
        
        
        
        The cumulants are 
        
        '''
        pass
    
if __name__ == "__main__":
    
    GeneralizedEdgeworthExpansion().model()