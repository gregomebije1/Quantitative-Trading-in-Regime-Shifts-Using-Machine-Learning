# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import ta # Technical Analysis library

# Define the symbols and the time period
symbols = ["AUDCAD", "EURCHF"]
start = "2006-01-01"
end = "2020-12-31"

# Fetch the data from Yahoo Finance
df = pd.DataFrame()
for s in symbols:
df[s] = web.DataReader(s, data_source="yahoo", start=start, end=end)["Adj Close"]

# Resample the data to 1-minute bars
df = df.resample("1T").ffill()

# Define the values for the grid
AUDCAD = [2, 2.5, 3, 3.5, 4]
entry_threshold = [0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.25, 1.5, 2, 2.5]
lookback = [30, 60, 90, 120, 180, 240, 360, 720]

# Create a 5 x 10 x 8 grid using numpy
grid = np.array(np.meshgrid(AUDCAD, entry_threshold, lookback)).T.reshape(-1,3)

# Convert the grid to a pandas dataframe
df_grid = pd.DataFrame(grid, columns=["AUDCAD", "entry_threshold", "lookback"])

# Define the technical indicators and the lookback windows
indicators = ["Z-score", "Money Flow", "Force Index", "Donchian Channel", "Average True Range", "Awesome Oscillator", "Average Directional Index"]
windows = [50,100,200,400,800,1600,3200]

# Loop over the indicators and the windows and calculate the values for each symbol and each minute bar
for ind in indicators:
for win in windows:
for sym in symbols:
# Use the ta library to compute the indicator values
if ind == "Z-score":
# Calculate the Bollinger bands z-score using a simple moving average and standard deviation
ma = df[sym].rolling(win).mean() # Moving average
std = df[sym].rolling(win).std() # Moving standard deviation
upper = ma + 2 * std # Upper band
lower = ma - 2 * std # Lower band
z_score = (df[sym] - ma) / (upper - lower) # Z-score
df[f"{ind}-{sym}({win})"] = z_score # Add to the dataframe

elif ind == "Money Flow":
# Calculate the money flow index using high, low and close prices and volume
high = web.DataReader(sym, data_source="yahoo", start=start, end=end)["High"] # High price
low = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Low"] # Low price
close = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Close"] # Close price
volume = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Volume"] # Volume
mfi = ta.volume.MFIIndicator(high=high.resample("1T").ffill(), low=low.resample("1T").ffill(), close=close.resample("1T").ffill(), volume=volume.resample("1T").ffill(), window=win) # Money flow index indicator object
df[f"{ind}-{sym}({win})"] = mfi.money_flow_index() # Add to the dataframe

elif ind == "Force Index":
# Calculate the force index using close price and volume
close = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Close"] # Close price
volume = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Volume"] # Volume
fi = ta.volume.ForceIndexIndicator(close=close.resample("1T").ffill(), volume=volume.resample("1T").ffill(), window=win) # Force index indicator object
df[f"{ind}-{sym}({win})"] = fi.force_index() # Add to the dataframe

elif ind == "Donchian Channel":
# Calculate the donchian channel indicator using high and low prices
high = web.DataReader(sym, data_source="yahoo", start=start, end=end)["High"] # High price
low = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Low"] # Low price
dc = ta.volatility.DonchianChannel(high=high.resample("1T").ffill(), low=low.resample("1T").ffill(), window=win) # Donchian channel indicator object
df[f"{ind}-{sym}({win})"] = dc.donchian_channel_pband() # Add to the dataframe

elif ind == "Average True Range":
# Calculate the average true range using high, low and close prices
high = web.DataReader(sym, data_source="yahoo", start=start, end=end)["High"] # High price
low = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Low"] # Low price
close = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Close"] # Close price
atr = ta.volatility.AverageTrueRange(high=high.resample("1T").ffill(), low=low.resample("1T").ffill(), close=close.resample("1T").ffill(), window=win) # Average true range indicator object
df[f"{ind}-{sym}({win})"] = atr.average_true_range() # Add to the dataframe

elif ind == "Awesome Oscillator":
# Calculate the awesome oscillator using high and low prices
high = web.DataReader(sym, data_source="yahoo", start=start, end=end)["High"] # High price
low = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Low"] # Low price
ao = ta.momentum.AwesomeOscillatorIndicator(high=high.resample("1T").ffill(), low=low.resample("1T").ffill(), window1=win//2, window2=win) # Awesome oscillator indicator object
df[f"{ind}-{sym}({win})"] = ao.awesome_oscillator() # Add to the dataframe

elif ind == "Average Directional Index":
# Calculate the average directional index using high, low and close prices
high = web.DataReader(sym, data_source="yahoo", start=start, end=end)["High"] # High price
low = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Low"] # Low price
close = web.DataReader(sym, data_source="yahoo", start=start, end=end)["Close"] # Close price
adx = ta.trend.ADXIndicator(high=high.resample("1T").ffill(), low=low.resample("1T").ffill(), close=close.resample("1T").ffill(), window=win) # Average directional index indicator object
df[f"{ind}-{sym}({win})"] = adx.adx() # Add to the dataframe

# Merge the grid and the technical indicators dataframes
df_merged = pd.merge(df_grid, df, left_index=True, right_index=True)

# Split the merged dataframe into train and test sets (80%/20%)
train_merged = df_merged.iloc[:train_size]
test_merged = df_merged.iloc[train_size:]

# Define a target variable for market regimes based on some criterion (e.g., volatility)
volatility = train_merged.std(axis=1) # Calculate the daily volatility of returns in the train set
threshold = volatility.quantile(0.8) # Set a threshold based on some quantile
target = (volatility > threshold).astype(int) # Label the days as 1 if volatility is above the threshold, 0 otherwise

# Train a random forest classifier with boosting to predict the market regimes using the daily returns and the technical indicators as features
rfc = RandomForestClassifier(n_estimators=100, random_state=42) # Random forest classifier object with 100 trees
abc = AdaBoostClassifier(base_estimator=rfc, n_estimators=50, random_state=42) # AdaBoost classifier object with 50 boosted trees
abc.fit(train_merged.drop(symbols, axis=1), target) # Fit the classifier to the train set features and the target variable

# Predict the market regimes for the test set using the trained classifier
test_regimes = abc.predict(test_merged.drop(symbols, axis=1)) # Predict the market regimes for each day in the test set using the classifier
test_merged["Regime"] = test_regimes # Add the regime labels to the test merged dataframe

# Define a function to find the optimal parameter combination for a given regime and market condition based on some metric (e.g., future one-day return)
def find_optimal_params(data, regime, metric):
# Filter the data by regime
data_regime = data[data["Regime"] == regime]

# Initialize the variables
optimal_params = None # Optimal parameter combination
