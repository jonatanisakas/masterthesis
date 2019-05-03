import pandas as pd
from readData import ReadData
from constantVol import ConstantVolatilityModel
from hestonModel import HestonModel
from jumpdifusion import JumpDiffusionModel

class MonteCarloSimulations():


    def __init__(self, df, rf, iterations, periods, longvol, gamma, voldrift, corr, lambda_j, mu_j, sigma_j, a, b):
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
        self.sigma_j = sigma_j
        self.a = a
        self.b = b
    
    #Main function
    def calculateOptionPrices(self):
        #Define the different list for the option prices
        OptionPriceNormalDist = []
        OptionPriceStocahsitcVol = []
        OptionPriceJumpDiffusion = []
        
        #Loop through all the rows in the dataframe
        for index, row in self.df.iterrows(): 
            
            self.S0, self.K, self.sigma, self.T =  self.df['Stock price'].iloc[index], self.df['Strike price'].iloc[index], self.df['Volatility'].iloc[index], self.df['T'].iloc[index]
            
            #append all the option prices to a list
            OptionPriceNormalDist.append(ConstantVolatilityModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods).callOptionPriceNormalDistribution())
            OptionPriceStocahsitcVol.append(HestonModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods, self.longvol, self.gamma, self.voldrift, self.corr).optionPricingStochasticVolatility())
            OptionPriceJumpDiffusion.append(JumpDiffusionModel(self.S0, self.K, self.sigma, self.T, self.rf, self.iterations, self.periods, self.lambda_j, self.mu_j, self.sigma_j, self.a, self.b).optionPricingJumpDiffusion())
        
        self.df['Option Price ND'] = OptionPriceNormalDist
        self.df['Option Price SD'] = OptionPriceStocahsitcVol
        self.df['Option Price JD'] = OptionPriceJumpDiffusion
        return self.df
    
if __name__ == "__main__":
    
    path = 'M:\Master thesis\code\Data\TestImportFile2.csv'
    header = None
    index_col = None
    df = pd.DataFrame(ReadData(path, header, index_col).readFile())
    
    #General
    rf=0.05
    iterations = 1000
    periods = 500
    
    #For Stochastic Volatility
    longvol = 0.35
    gamma = 0.25
    voldrift = 2
    corr = -0.75
    
    #For JumpDiffusion
    lambda_j = 0.20
    mu_j = 0.05
    sigma_j = 0.3
    a = 0.15
    b = 0.15
    model = MonteCarloSimulations(df, rf, iterations, periods, longvol, gamma, voldrift, corr, lambda_j, mu_j, sigma_j, a, b)
    print(model.calculateOptionPrices())