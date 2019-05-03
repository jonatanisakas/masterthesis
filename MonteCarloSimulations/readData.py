import pandas as pd


class ReadData():
    
    def __init__(self, path, header=0, index_col=None):
        
        self.path = path
        self.ending = path.split('.')[1]
        self.header = header
        self.index_col = index_col
    
    def readFile(self):
        if self.ending == 'csv':
            df =  self.readCsv()
            return df
        else: 
            df = self.readExcel()
            return df
            
    def readCsv(self):
        self.df = pd.read_csv(self.path, self.header, self.index_col, engine='python')
        self.df['T'] = self.calculateTimeToMaturity()
        
        return self.df
        
    def readExcel(self):
        self.df = pd.read_excel(self.path, self.header, self.index_col, engine='python')
        self.df['T'] = self.calculateTimeToMaturity()
        
        return self.df
        
        #Extact Time to Maturity
    def calculateTimeToMaturity(self):
        date = pd.to_datetime(self.df['Date'], dayfirst=True, errors='coerce')
        Exp = pd.to_datetime(self.df['Expiration'], dayfirst=True, errors='coerce')
        T = (Exp - date)/365
        T = (T.dt.total_seconds())/(60*60*24)
        return T
        
        
if __name__ == "__main__":
    
    path = 'M:\Master thesis\code\Data\TestImportFile.xlsx'
    header = 0
    index_col = None
    ReadData(path, header, index_col).readFile()