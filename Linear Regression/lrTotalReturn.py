from math import dist
import numpy as np
import pandas as pd
from plotly import data
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

def lnFit(CSVfile, columnName, outputFile):
    df = pd.read_csv(CSVfile)
    df['Log-Data'] = np.log(df[columnName])
    df = df.drop(columns=['Unnamed: 0'])
    df.to_csv(outputFile)
    return df
    
def modelQuality(CSVfile, columnName): # Only tests exponential, gamma, log norm, weibull, and invgauss
    df = pd.read_csv(CSVfile)
    data = df[columnName]
    n = len(data)
    results=[]

    distributions = ['norm', 't', 'laplace', 'johnsonsu', 'genhyperbolic', 'hypsecant']
    for dist in distributions:
        try:
            distr = getattr(stats, dist)
            params = distr.fit(data, floc=0)  # Fit the distribution to the data and floc=0 to force location parameter to be zero
            log_likelihood = np.sum(distr.logpdf(data, *params))  # Calculate log-likelihood, *param takes list/tuple and unpack its contents into individual arguments for the function
            k = len(params)

            aic = 2*k - 2*log_likelihood
            bic = k*np.log(n) - 2*log_likelihood

            results.append(
                    {
                        "Distribution": dist, 
                        "AIC": aic, 
                        "BIC": bic,
                        "Params": params
                    }
                )
            
        except Exception as e:
            print(f"Error fitting {dist}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="AIC")
    return results_df

def stockDNA(CSVfile, columnName):
    df = pd.read_csv(CSVfile)
    data = df[columnName].dropna() # Drop any NaN values from the data
    a, b, loc, scale = stats.johnsonsu.fit(data)
    annualReturn = loc * 252
    annualVolatility = scale * np.sqrt(252)
    sharpeRatio = (annualReturn - 0.05) / annualVolatility # Assuming a risk-free rate of 5%
    

    # From gemini:
    # The Sharpe ratio measures an investment's risk-adjusted return by calculating'
    # 'the average return earned in excess of the risk-free rate per unit of volatility(standard deviation).'
    # 'It helps investors determine if a portfolio's higher returns are due to smart
    # investment decisions or taking on excessive, higher-risk volatility.

    paramList = []
    paramList.append({
        "Distribution": "Johnson SU", 
        "Skew_Shape(a)": a,    # Gamma
        "Tail_Shape(b)": b,    # Delta
        "Location(μ)": loc,    # Mean-ish
        "Scale(σ)": scale,      # Volatility
        "Annual Volatility": annualVolatility,
        "Annual Return": annualReturn,
        "Sharpe Ratio": sharpeRatio
    })

    var95 = stats.johnsonsu.ppf(0.05, a, b, loc, scale) # Value at risk tells you the maximum expected loss over a specific timeframe with 95% confidence
    print(f"95% Daily VaR: {var95:.2f}%")

    results_df = pd.DataFrame(paramList)
    results_df = results_df.round(4)
    return a, b, loc, scale
    
def getStockData(tickerSymbol, outputFile):
    ticker = yf.Ticker(tickerSymbol)
    df = ticker.history(start="2016-03-11", end="2026-03-11")
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d") # Convert datetime to string format
    df['Year-Month'] = df['Date'].str.split(" ").str[0].str[0:7] # New Column of Year and Month
    
    df['Simple Returns'] = df['Close'].pct_change() * 100
    df = df.dropna(subset=['Simple Returns'])

    df.to_csv(outputFile, index=False)

def compareToSPY(CSVfile1, CSVfile2, nameOfStock, nameOfStock2):
    df1 = pd.read_csv(CSVfile1)
    df2 = pd.read_csv(CSVfile2)

    DNA1 = stockDNA(CSVfile1, "Simple Returns")
    DNA2 = stockDNA(CSVfile2, "Simple Returns")

    DNA1['Ticker'] = nameOfStock
    DNA2['Ticker'] = nameOfStock2

    comparison_df = pd.concat([DNA1, DNA2]).set_index('Ticker')
    print(comparison_df)

def predictEnd2026(currentPrice, a, b, loc, scale, days=252, simulations=10000):    
    simReturns = stats.johnsonsu.rvs(a, b, loc, scale, size=(days, simulations))
    priceMultipliers = 1 + (simReturns / 100)
    pricePaths = currentPrice * priceMultipliers.cumprod(axis=0)
    
    fig = go.Figure()

    for i in range(min(simulations, 100)):
        fig.add_trace(go.Scatter(
            y=pricePaths[:, i], 
        mode='lines', 
        line=dict(width=1, color='blue'),
        showlegend=False
    ))
        
    fig.add_hline(
    y=currentPrice, 
    line_dash="dash", 
    line_color="red", 
    annotation_text="Starting Price", 
    annotation_position="bottom right"
    )

    fig.update_layout(
    title="Monte Carlo: 100 Potential Paths (1 Year)",
    xaxis_title="Days",
    yaxis_title="Price ($)",
    template="plotly_dark",
    hovermode="x"
    )

    fig.show()
    final_prices = pricePaths[-1]
    hist_fig = px.histogram(
        final_prices, 
        nbins=50, 
        title="Distribution of Predicted Ending Prices",
        labels={'value': 'Final Price ($)'},
        template="plotly_dark"
    )
    hist_fig.show()


    return pricePaths[-1]  # Return the final prices at the end of the simulation
