import pandas as pd
import numpy as np

def readData(path):
    
    path_read = path + '.csv'

    df = pd.read_csv(path_read, index_col=0, engine='python')
    
    
    #General
    S = df['Stock price']
    K = df['Strike price']
    r = 0.01205
    T = df['T']
    
    #print(df['Call Price ND'])

    for index, row in df.iterrows(): 
        
        if K.iloc[index] < S.iloc[index]:
            
            #Normal distribution
            c_price = df['Put Price ND'].iloc[index] + S.iloc[index] - K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Call Price ND'].iloc[index] = c_price
            df['Implied Volatility call ND'] = df['Implied Volatility put ND']
            
            #Jump Diffusion
            c_price = df['Put Price JD'].iloc[index] + S.iloc[index] - K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Call Price JD'].iloc[index] = c_price
            df['Implied Volatility call JD'] = df['Implied Volatility put JD']
            
            
            #Stochastic Volatility
            c_price = df['Put Price SD'].iloc[index] + S.iloc[index] - K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Call Price SD'].iloc[index] = c_price
            df['Implied Volatility call SD'] = df['Implied Volatility put SD']
            
            
            #SVJD
            c_price = df['Put Price SVJD'].iloc[index] + S.iloc[index] - K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Call Price SVJD'].iloc[index] = c_price
            df['Implied Volatility call SVJD'] = df['Implied Volatility put SVJD']
            
            
        if K.iloc[index] >= S.iloc[index]:
            
            #Normal distribution
            p_price = df['Call Price ND'].iloc[index] - S.iloc[index] + K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Put Price ND'].iloc[index] = p_price
            df['Implied Volatility put ND'] = df['Implied Volatility call ND']
            
            #Jump Diffusion
            p_price = df['Call Price JD'].iloc[index] - S.iloc[index] + K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Put Price JD'].iloc[index] = p_price
            df['Implied Volatility put JD'] = df['Implied Volatility call JD']
            
            #Stochastic Volatility
            p_price = df['Call Price SD'].iloc[index] - S.iloc[index] + K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Put Price SD'].iloc[index] = p_price
            df['Implied Volatility put SD'] = df['Implied Volatility call SD']
            
            #SVJD
            p_price = df['Call Price SVJD'].iloc[index] - S.iloc[index] + K.iloc[index] *np.exp(-r*T.iloc[index])
            df['Put Price SVJD'].iloc[index] = p_price
            df['Implied Volatility put SVJD'] = df['Implied Volatility call SVJD']

    
    pathNew = path + '_updated.csv'
    
    df.to_csv(pathNew)
            
            
            
    print( df['Call Price ND'])
readData('M:/Master thesis/Data/Results/test')