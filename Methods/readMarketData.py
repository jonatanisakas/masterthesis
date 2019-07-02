import pandas as pd
import numpy as np
import datetime


class ReadData():
    
    def __init__(self, path, pathEquity, pathRiskFree, exportPath, openInterest, header=0, index_col=None):
        
        self.path = path
        self.ending = path.split('.')[1]
        self.header = header
        self.index_col = index_col
        self.pathEquity = pathEquity
        self.interest = openInterest
        self.exportPath = exportPath
        self.pathRiskFree = pathRiskFree

        
    def readFile(self):
        if self.ending == 'csv':
            df =  self.readCsv()
            df.to_csv(self.exportPath, index=False, sep=',')
            return df
        else:
            print ("error, file not CSV")
            
    def readCsv(self):
        self.df = pd.read_csv(self.path, self.index_col, engine='python')
        self.df = self.df.dropna()
        self.equity, self.riskFree = self.readEquityPricesAndRiskFree()
        
        #Clean data
        self.df = self.cleanData()
        return self.df
        
        
    def readEquityPricesAndRiskFree(self):
        #Read both equity and RiskFree rate into Pandas Dataframes
        dfEquity = pd.read_csv(self.pathEquity, engine='python')
        riskFree = pd.read_csv(self.pathRiskFree, engine='python')
        equity = pd.DataFrame()
        equity['date'], equity['close'] = dfEquity['caldt'], dfEquity['spindx']
        #Convert to precentage
        riskFree['rate'] = riskFree['rate']/100
        
        return equity, riskFree
    
    def cleanData(self):
        
        #Remove contracts with low interest
        self.df = self.df[self.df['open_interest'] > self.interest]
        
        #This is a complete mess, but the date format from WRDS is completley different and needed 
        #a lot of tweeking to fit the later format
        
        #Create a new column
        self.riskFree['newdate'] = np.nan
        #Split the date string to three items in a list
        self.riskFree['date'] = self.riskFree['date'].str.split('/')\
        #create three new columns designated to the year, monht and day
        self.riskFree['year'], self.riskFree['month'], self.riskFree['day'] = self.riskFree['date'].str[2], self.riskFree['date'].str[0], self.riskFree['date'].str[1]
        #add zeros to dates and months that are shorter than 2
        self.riskFree['month'] = self.riskFree['month'].apply(lambda x: '{0:0>2}'.format(x))
        
        self.riskFree['day'] = self.riskFree['day'].apply(lambda x: '{0:0>2}'.format(x))
        #append the year month and date in the correct order
        self.riskFree['newdate'] = self.riskFree['year'] + self.riskFree['month'] + self.riskFree['day']
        #replace the old column with the new date and make it an integer
        self.riskFree['date'] = self.riskFree['newdate'].astype(int)
    
        
        #Add the equity price
        self.df['stock'] = np.nan
        for index, row in self.equity.iterrows():
            
            date = self.equity['date'].iloc[index]
            price = self.equity['close'].iloc[index]
            
            self.df['stock'] = np.where((self.df['date'] == date), price, self.df['stock'])
        
        #Add risk-free rate
        self.df['riskfree'] = np.nan
        for index, row in self.riskFree.iterrows():
        
            
            date = self.riskFree['date'].iloc[index]
            price = self.riskFree['rate'].iloc[index]
            
            self.df['riskfree'] = np.where((self.df['date'] == date), price, self.df['riskfree'])
            
        #Add option contract which is the average of the big and ask price
        self.df['option_price'] = (self.df['best_bid'] + self.df['best_offer'])/2
        
        #Strike price 
        self.df['strike_price'] = self.df['strike_price'] /1000
        
        #Moneyness
        self.df['moneyness'] = self.df['strike_price']/self.df['stock']
        
        #Calculate time to maturity
        self.df['T'] = self.calculateTimeToMaturity()
        
        #Drop unecessary columns
        del self.df['optionid'], self.df['index_flag'], self.df['issuer'], self.df['exercise_style']
            
        return self.df

        

        #Extact Time to Maturity
    def calculateTimeToMaturity(self):
        date = pd.to_datetime(self.df['date'], format='%Y%m%d', utc=True, dayfirst=False, errors='coerce')
        Exp = pd.to_datetime(self.df['exdate'], format='%Y%m%d', utc=True, dayfirst=False, errors='coerce')
        T = (Exp - date)/365
        T = (T.dt.total_seconds())/(60*60*24)
        return T
        
if __name__ == "__main__":
    
    path = 'M:\Master thesis\Data\sampleData.csv'
    pathEquity = 'M:\Master thesis\Data\s&p500.csv'
    pathRiskFree = 'M:/Master thesis/Data/risk-free.csv'
    exportPath = 'M:\Master thesis\Data\cleanOptionData.csv'
    header = 0
    index_col = None
    openInterest = 20
    ReadData(path, pathEquity, pathRiskFree, exportPath, header, openInterest, index_col).readFile()