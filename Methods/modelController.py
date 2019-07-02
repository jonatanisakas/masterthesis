import pandas as pd
#from gaussianMixtureModel import GaussianMixtureModel
from optionpricing import OptionPricing


class ModelController():
    
    def __init__(self, csvPath, indexCol, n_samples, time_to_maturity, underlying_stock, risk_free, bandwidth, mixture_models):
        
        self.csvPath = csvPath
        self.indexCol = indexCol
        self.n_samples = n_samples
        self.T = time_to_maturity
        self.S = underlying_stock
        self.r = risk_free
        self.bandwidth = bandwidth
        self.mixture_models = mixture_models
        
    def getData(self):
        
        df = pd.read_csv(self.csvPath, self.indexCol, engine='python', error_bad_lines=False )
        df = df.dropna()
        return df
        
    def runModels(self):

        OptionPricing(self.getData(), self.n_samples, self.T, self.S, self.r, self.bandwidth, self.mixture_models).functionEstimationKernelRegression()
        OptionPricing(self.getData(), self.n_samples, self.T, self.S, self.r, self.bandwidth, self.mixture_models).functionMixtureModel()
        
    
if __name__ == "__main__":
    
    csvPath = 'M:/Master thesis/Data/test4.csv'
    indexCol = None
    n_samples = 20000
    time_to_maturity = 1
    underlying_stock = 2430.06
    risk_free = 0.012053606
    
    #Bandwidth selection for kernel regression: bw_scott, bw_silverman, or None
    #bandwidth = 'bw_scott'
    bandwidth = None
    mixture_models = 2
    
    ModelController(csvPath, indexCol, n_samples, time_to_maturity, underlying_stock, risk_free, bandwidth, mixture_models).runModels()