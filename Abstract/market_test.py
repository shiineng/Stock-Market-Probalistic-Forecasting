from math import dist
import numpy as np
import pandas as pd
from plotly import data
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf

def overall_analysis(ticker_symbol, start_date, end_date, return_column, buy_price, days, simulations, line_color, hist_color):
    df = get_stock(ticker_symbol, start_date, end_date, return_column)
    aic_test = model_quality(df)
    print(f"AIC TEST:\n{aic_test}")
    # print(f"BIC TEST: \n {bic_test}")
    plot_line_fit(df, ticker_symbol)
    params, best_dist = get_stock_dna(df, aic_test, 0.05)
    predict_by_input(buy_price, best_dist, params, days, simulations, ticker_symbol, line_color, hist_color)




"""
I suggest putting start date one day before you're actual desired start date
Dates should be "YYYY-MM-DD"
"""
def get_stock(ticker_symbol, start_date, end_date, return_column):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start = start_date, end = end_date)
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d") # Converts date to string
    df["Year-Month"] = df["Date"].str.split(" ").str[0].str[0:7] # New column of YYYY-MM to use for monthly averages

    df["Simple-Returns"] = df[return_column].pct_change() * 100
    df = df.dropna(subset=["Simple-Returns"]) # Why I suggest starting one day before

    return df

def model_quality(df, column_of_interest="Simple-Returns"):
    n = len(df[column_of_interest])
    results = []
    aic_values = []

    distributions = ['norm', 't', 'laplace', 'johnsonsu', 'genhyperbolic', 'hypsecant', 'skewnorm', 'cauchy']
    for dist in distributions:
        try:
            distr = getattr(stats, dist)
            params = distr.fit(df[column_of_interest], floc=0)
            log_likelihood = np.sum(distr.logpdf(df[column_of_interest], *params))
            k = len(params)

            aic = 2*k - 2*log_likelihood
            bic = k*np.log(n) - 2*log_likelihood
            
            results.append(
                {
                    "Distribution:": dist,
                    "AIC": aic,
                    "BIC": bic,
                    "Params": params,
                    "Param Quantity": k,
                }
            )

        except Exception as e:
            print(f"Error fitting {dist}: e")


    results_df_aic = pd.DataFrame(results).sort_values(by="AIC")
    # results_df_bic = pd.DataFrame(results).sort_values(by="BIC")

    return results_df_aic #, results_df_bic

def plot_line_fit(df, ticker_symbol, column_of_interest="Simple-Returns"):
    df[column_of_interest] = df[column_of_interest].replace([np.inf, -np.inf], np.nan).dropna()

    x_range = np.linspace(df[column_of_interest].min(), df[column_of_interest].max(), 500)
    pdf_val = ['norm', 't', 'laplace', 'johnsonsu', 'genhyperbolic', 'hypsecant', 'skewnorm', 'cauchy']
    fig = px.histogram(
        df[column_of_interest],
        nbins=100,
        histnorm="probability density",
        title=f'Distribution of Daily Returns: {ticker_symbol}',
        labels={'value: ' 'Daily Return (%)'},
        opacity=0.6,
        template="plotly_dark"
    )

    for dist in pdf_val:
        try:
            distr = getattr(stats, dist)
            params = distr.fit(df[column_of_interest])
            y_val = distr.pdf(x_range, *params)

            fig.add_scatter(x=x_range, y=y_val, mode='lines', name=f'{dist} PDF')
        except Exception as e:
            print(f"Error with {dist}: {e}")
        
    fig.show()

def get_stock_dna(df, aic_test, rf, column_of_interest="Simple-Returns"):
    df[column_of_interest] = df[column_of_interest].replace([np.inf, -np.inf], np.nan).dropna()
    best_dist = aic_test.iloc[0]["Distribution:"]
    best_row = aic_test.iloc[0]
    params = best_row["Params"]
    loc = params[-2]
    scale = params[-1]


    if best_dist == 'johnsonsu':
        a, b, loc, scale = stats.johnsonsu.fit(df[column_of_interest])

    elif best_dist == 'norm':
        loc, scale = stats.norm.fit(df[column_of_interest])
        
    elif best_dist == 't':
        df_param, loc, scale = stats.t.fit(df[column_of_interest])
        
    elif best_dist == 'laplace':
        loc, scale = stats.laplace.fit(df[column_of_interest])
        
    elif best_dist == 'genhyperbolic':
        p, a, b, loc, scale = stats.genhyperbolic.fit(df[column_of_interest])
        
    elif best_dist == 'hypsecant':
        loc, scale = stats.hypsecant.fit(df[column_of_interest])
        
    elif best_dist == 'skewnorm':
        a, loc, scale = stats.skewnorm.fit(df[column_of_interest])
        
    else: # cauchy
        loc, scale = stats.cauchy.fit(df[column_of_interest])

    annual_return = loc * 252 # Std trading period is 252
    annual_volatility = scale * np.sqrt(252) # Std trading period is 252
    sharpe_ratio = (annual_return - rf) / annual_volatility

    print(f"Best Distribution: {best_dist}")
    print(f"Annual Return: {annual_return:.2f}% | Annual Vol: {annual_volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    return params, best_dist

def predict_by_input(buy_price, best_dist, params, days, simulations, ticker_symbol, line_color, hist_color):
    dist_obj = getattr(stats, best_dist)
    sim_returns = dist_obj.rvs(*params, size=(days, simulations))
    price_multipliers = 1 + (sim_returns / 100)
    price_paths = buy_price * price_multipliers.cumprod(axis=0)

    fig = go.Figure()

    for sim in range(min(simulations, 100)):
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=price_paths[:, sim],
            mode='lines',
            line=dict(width=1, color=line_color),
            showlegend=False
        ))
    
    h_color = "white" if "255" in line_color else "red"
    
    fig.add_hline(
        y=buy_price, 
        line_dash="dash", 
        line_color=h_color, 
        annotation_text="Starting Price", 
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=f"Monte Carlo: 100 Potential Paths [{ticker_symbol}]",
        xaxis_title="Days",
        yaxis_title="Price ($)",
        template="plotly_dark",
        hovermode="x"
    )

    fig.update_xaxes(range=[0, days])
    fig.show()

    final_prices = price_paths[-1]
    hist_fig = px.histogram(
        final_prices, 
        nbins=50, 
        title=f"Distribution of Predicted Ending Prices [{ticker_symbol}]",
        labels={'value': 'Final Price ($)'},
        template="plotly_dark",
        color_discrete_sequence=[hist_color]
    )
    hist_fig.show()

    positiveCount = np.sum(final_prices >= buy_price)
    negativeCount = simulations - positiveCount
    
    print(f"\n--- {ticker_symbol} Monte Carlo Results ---")
    print(f"Prob. of positive or neutral return: {round(positiveCount/simulations, 3)}")
    print(f"Prob. of negative return: {round(negativeCount/simulations, 3)}")
    return final_prices