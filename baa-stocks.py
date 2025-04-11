import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])
    
df = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = df['Volatility'].values[1:]
div = df['Dividends'].values[1:]
price = df['Price'].values
baa = df['BAA'].values
df2 = pd.read_excel('century.xlsx', sheet_name = 'earnings')
cpi = df2['CPI'].values[9:]
N = len(vol)
inflation = np.diff(np.log(cpi))
nominal = np.array([np.log(price[k+1] + div[k]) - np.log(price[k]) for k in range(N)])
real = nominal - inflation
normReal = real/vol
normNominal = nominal/vol

regDF1 = pd.DataFrame({'const' : 1, 'rate' : baa[:-1], 'duration' : np.diff(baa)})
regDF2 = pd.DataFrame({'const' : 1/vol, 'rate' : baa[:-1]/vol, 'duration' : np.diff(baa)/vol, 'vol' : 1})

nREG = OLS(normNominal, regDF1).fit()
print(nREG.summary())
nres = nREG.resid
plots(nres, 'nominal-1')
analysis(nres, 'nominal-1')

rREG = OLS(normReal, regDF1).fit()
print(rREG.summary())
rres = rREG.resid
plots(rres, 'real-1')
analysis(rres, 'real-1')

nREG = OLS(normNominal, regDF2).fit()
print(nREG.summary())
nres = nREG.resid
plots(nres, 'nominal-2')
analysis(nres, 'nominal-2')

rREG = OLS(normReal, regDF2).fit()
print(rREG.summary())
rres = rREG.resid
plots(rres, 'real-2')
analysis(rres, 'real-2')