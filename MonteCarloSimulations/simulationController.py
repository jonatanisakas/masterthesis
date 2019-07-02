import pandas as pd
import time as time


from readData import ReadData
from constantVol import ConstantVolatilityModel
from hestonModel import HestonModel
from jumpDiffusion import JumpDiffusionModel
from stochasticVolandStochasticJump import StochasticVolandStochasticJumpModel
from impliedvolatility import ImpliedVolatility

start_time = time.clock()

class MonteCarloSimulations():


    def __init__(self, df, rf, iterations, periods, longvol, gamma, voldrift, corr, lambda_j, mu_j, delta, tick):
        self.df = df
        self.rf = rf
        self.iterations = iterations #Stock Price paths
        self.periods = periods #Number of time steps
        self.longvol = longvol
        self.gamma = gamma
        self.voldrift = voldrift
        self.corr = corr
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.delta = delta
        self.tick = tick
    
    #Main function
    def calculateOptionPrices(self):
        #Define the different list for the option prices
        callNormalDist = []
        callStocahsitcVol = []
        callJumpDiffusion = []
        callStoVolStoJump = []
        
        putNormalDist = []
        putStocahsitcVol = []
        putJumpDiffusion = []
        putStoVolStoJump = []
        
        impVolBSCall = []
        impVolSVCall = []
        impVolJDCall = []
        impVolSVJDCall = []
        
        impVolBSPut = []
        impVolSVPut = []
        impVolJDPut = []
        impVolSVJDPut = []
        p = self.df['Strike price']
        leniter = len(p)-1
        
        #Loop through all the rows in the dataframe
        for index, row in self.df.iterrows(): 
            
            self.S0, self.K, self.sigma, self.T =  self.df['Stock price'].iloc[index], self.df['Strike price'].iloc[index], self.df['Volatility'].iloc[index], self.df['T'].iloc[index]
            print(index, ' out of ', leniter, ', youre doing great!')
            #append all the option prices to a list
            #Black-Scholes
            option_call_BS, option_put_BS = ConstantVolatilityModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations,\
                                                                self.periods, self.tick).callOptionPriceNormalDistribution()
            imp_vol_call_BS, imp_vol_put_BS = ImpliedVolatility(self.S0, self.K, self.T, self.rf, self.sigma).imp_vol(option_call_BS, option_put_BS)
            callNormalDist.append(option_call_BS)
            putNormalDist.append(option_put_BS)
            impVolBSCall.append(imp_vol_call_BS)
            impVolBSPut.append(imp_vol_put_BS)
            
            #Stochastic Volatility, Heston Model
            option_call_SV, option_put_SV = HestonModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods,\
                                                                self.longvol, self.gamma, self.voldrift, self.corr, self.tick).optionPricingStochasticVolatility()
            imp_vol_call_SV, imp_vol_put_SV = ImpliedVolatility(self.S0, self.K, self.T, self.rf, self.sigma).imp_vol(option_call_SV, option_put_SV)
            callStocahsitcVol.append(option_call_SV)
            putStocahsitcVol.append(option_put_SV)
            impVolSVCall.append(imp_vol_call_SV)
            impVolSVPut.append(imp_vol_put_SV)
            
            #Jump Difusion
            option_call_JD, option_put_JD = JumpDiffusionModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods,\
                                                                self.lambda_j, self.mu_j, self.delta, self.tick).optionPricingJumpDiffusion()
            imp_vol_call_JD, imp_vol_put_JD = ImpliedVolatility(self.S0, self.K, self.T, self.rf, self.sigma).imp_vol(option_call_JD, option_put_JD)
            callJumpDiffusion.append(option_call_JD)
            putJumpDiffusion.append(option_put_JD)
            impVolJDCall.append(imp_vol_call_JD)
            impVolJDPut.append(imp_vol_put_JD)
            
            
            
            #Stochastic Volatility and Jump difusion
            option_call_SVJD, option_put_SVJD = StochasticVolandStochasticJumpModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods,\
                                                                self.longvol, self.gamma, self.voldrift, self.corr, self.lambda_j, self.mu_j, self.delta, self.tick).optionPricingStochasticVolandJump()
            imp_vol_call_SVJD, imp_vol_put_SVJD = ImpliedVolatility(self.S0, self.K, self.T, self.rf, self.sigma).imp_vol(option_call_SVJD, option_put_SVJD)
            callStoVolStoJump.append(option_call_SVJD)
            putStoVolStoJump.append(option_put_SVJD)
            impVolSVJDCall.append(imp_vol_call_SVJD)
            impVolSVJDPut.append(imp_vol_put_SVJD)
            
            
        self.df['Call Price ND'], self.df['Put Price ND'], self.df['Implied Volatility call ND'], self.df['Implied Volatility put ND'] = callNormalDist, putNormalDist, impVolBSCall, impVolBSPut
        self.df['Call Price SD'], self.df['Put Price SD'], self.df['Implied Volatility call SD'], self.df['Implied Volatility put SD'] = callStocahsitcVol, putStocahsitcVol, impVolSVCall, impVolSVPut
        self.df['Call Price JD'], self.df['Put Price JD'], self.df['Implied Volatility call JD'], self.df['Implied Volatility put JD'] = callJumpDiffusion, putJumpDiffusion, impVolJDCall, impVolJDPut
        self.df['Call Price SVJD'], self.df['Put Price SVJD'], self.df['Implied Volatility call SVJD'], self.df['Implied Volatility put SVJD'] = callStoVolStoJump, putStoVolStoJump, impVolSVJDCall, impVolSVJDPut
        
        
        self.df.to_csv('M:/Master thesis/Data/splitdata/results/6000.csv')
        print ('Estimation time: ', time.clock() - start_time, "seconds")
        return self.df
    
if __name__ == "__main__":
    
    path = 'M:/Master thesis/Data/splitdata/simulationparameters_1000.csv'
    header = None
    index_col = None
    df = pd.DataFrame(ReadData(path, header, index_col).readFile())
    
    #General
    rf = 0.012986
    iterations = 100000
    periods = 200
    tick = 0.025 #1/2 of the smalles tick stize on the spx
    
    #For Stochastic Volatility
    longvol = 0.10
    gamma = 0.5
    kappa = 3.5
    rho = -0.7
    
    #For JumpDiffusion
    lambda_j = 0.5 #Jump frequency
    mu_j = -0.03 #Expected jump size
    delta = 0.15 #Jump size volatility
    
    model = MonteCarloSimulations(df, rf, iterations, periods, longvol, gamma, kappa, rho, lambda_j, mu_j, delta, tick)
    print(model.calculateOptionPrices())