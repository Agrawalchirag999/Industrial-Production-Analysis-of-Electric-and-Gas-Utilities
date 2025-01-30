# Industrial Production Analysis of Electric and Gas Utilities

## Overview
This project focuses on the analysis of industrial production data for electric and gas utilities using Python. The analysis includes time series decomposition, ARIMA modeling, and forecasting, providing insights into production trends and seasonal patterns.

## Requirements
To run this project, ensure you have the following Python packages installed:

- pandas
- numpy
- statsmodels
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy statsmodels matplotlib seaborn
```

## Data Preparation
1. **Load the Dataset**: The dataset is loaded from a CSV file.
   ```python
   df = pd.read_csv(r"C:\Users\chira\OneDrive\Desktop\SEM 5\ATSA\Project\IPG2211A2N.csv")
   ```
2. **Set Date as Index**: Convert the 'DATE' column to a datetime index and drop it from the DataFrame.
   ```python
   df.index = pd.to_datetime(df.DATE)
   df = df.drop(['DATE'], axis=1)
   ```

3. **Rename Columns**: Rename the production column for clarity.
   ```python
   df.columns = ['Production']
   ```

## Data Visualization
- **Plot Production Data**: Visualize the production data over time.
  ```python
  df.plot(figsize=(20,10), linewidth=2, fontsize=16)
  plt.xlabel('Year', fontsize=18)
  ```

- **Seasonal Decomposition**: Decompose the time series into trend, seasonality, and residuals.
  ```python
  decomposition = seasonal_decompose(df, model='additive')
  fig = decomposition.plot()
  ```

## Statistical Tests
- **ADF Test**: Conduct the Augmented Dickey-Fuller test to check for stationarity.
  ```python
  result = adfuller(df['Production'])
  print('ADF Statistic:', result[0])
  print('p-value:', result[1])
  ```

## ARIMA Modeling
1. **Model Fitting**: Fit an ARIMA model to the data.
   ```python
   model = ARIMA(df, order=(5,1,0))
   results = model.fit()
   print(results.summary())
   ```

2. **Parameter Selection**: Use grid search to find optimal parameters for SARIMA.
   ```python
   p = d = q = range(0, 2)
   pdq = list(itertools.product(p, d, q))
   seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
   ```

3. **Forecasting**: Generate predictions and visualize them.
   ```python
   pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
   ax = df['2013':].plot(label='observed')
   pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
   ```

## Evaluation Metrics
Calculate and print the Mean Squared Error (MSE) of the model's predictions.
```python
mse = round((SS_R/(n-1)),2)
print('Mean squared error:', mse)
```

## Conclusion
This project provides a comprehensive analysis of industrial production data using time series techniques in Python. Through visualization, statistical testing, and modeling, we gain valuable insights into production trends and forecasting capabilities.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


