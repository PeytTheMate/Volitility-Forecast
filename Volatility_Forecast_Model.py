# Modules
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt



def get_data(ticker, start_date, end_date):
    """Gets historical stock data for a given ticker, date range, and returns it as a dataframe."""
    data = yf.download(ticker, start=start_date, end=end_date) 
    data["Returns"] = data["Adj Close"].pct_change()
    return data.dropna()


def fit_garch(returns):
    """Fits a GARCH(1,1) model to the return series.

    Parameters:
    -----------
    returns : pd.Series
        Series of asset returns

    Returns:
    --------
    arch_model
        Fitted GARCH model object"""

    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    return results


def forecast_volatility(model, horizon=0):
    """Forecasts volatility using the fitted GARCH model.

    Parameters:
    -----------
    model : arch_model
        Fitted GARCH model
    horizon : int, optional (default=10)
        Number of days ahead to forecast

    Returns:
    --------
    np.ndarray
        Array of volatility forecasts for specified horizon
    """
    forecast = model.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.values[-1,:])


def evaluate(actual, predicted):
    """ Evaluates model performance using RMSE and MAE metrics.

    Parameters:
    -----------
    actual : np.ndarray
        Array of actual volatility values
    predicted : np.ndarray
        Array of predicted volatility values

    Returns:
    --------
    dict
        Dictionary containing RMSE and MAE values
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return {"RSME":rmse, "MAE": mae} 


def sensitivity_analysis(returns, p_range, q_range):
    """Performs sensitivity analysis on GARCH model parameters.

    Parameters:
    -----------
    returns : pd.Series
        Series of asset returns
    p_range : range
        Range of p values to test (ARCH terms)
    q_range : range
        Range of q values to test (GARCH terms)

    Returns:
    --------
    dict
        Dictionary with (p,q) tuples as keys and AIC values as values
    
    Example:
    --------
    >>> sensitivity = sensitivity_analysis(data['Returns'], range(1,4), range(1,4))
    """
    results = {}
    for p in p_range:
        for q in q_range:
            model = arch_model(returns, vol="Garch", p=p, q=q)
            fitted = model.fit(disp='off')
            results[(p, q)] = fitted.aic
    return results


def plot_volatility_forecast(data):
    """Plots realized vs predicted volatility.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing 'Realized_Volatility' and 'Predicted_Volatility' columns

    Returns:
    --------
    None
        Displays a matplotlib plot"""
    
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Realized_Volatility'], label='Realized Volatility')
    plt.plot(data.index, data['Predicted_Volatility'], label='Predicted Volatility')
    plt.title('Volatility Forecast vs Realized Volatility')
    plt.legend()
    plt.show()

# Main
if __name__ == "__main__":
    # Set parameters
    ticker = 'AAPL'  # Can change this to any stock
    start_date = '2020-01-01'
    end_date = '2023-12-31'

    # Get data
    data = get_data(ticker, start_date, end_date)
    
    # Fit GARCH model
    garch_model = fit_garch(data['Returns'])
    
    # Calculate volatilities
    data['Predicted_Volatility'] = garch_model.conditional_volatility
    data['Realized_Volatility'] = data['Returns'].rolling(window=22).std() * np.sqrt(252)
    
    # Make future predictions
    future_vol = forecast_volatility(garch_model)
    
    # Evaluate model
    evaluation_results = evaluate(data['Realized_Volatility'].dropna(), 
                                data['Predicted_Volatility'].dropna())
    
    # Run sensitivity analysis
    sensitivity_results = sensitivity_analysis(data['Returns'], range(1,4), range(1,4))
    
    # Plot results
    plot_volatility_forecast(data)
    
    # Print summary
    print(f"""
    Volatility Forecast Model Results for {ticker}:
    =============================================
    Evaluation Metrics:
    RMSE: {evaluation_results['RMSE']:.4f}
    MAE: {evaluation_results['MAE']:.4f}
    
    Future Volatility Forecast (next 10 days):
    {future_vol}
    
    Best GARCH Parameters:
    {min(sensitivity_results.items(), key=lambda x: x[1])}
    """)
