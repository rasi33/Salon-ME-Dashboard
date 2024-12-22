import pandas as pd

# Example data for forecast_data
data = {
	'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
	'forecast': [10, 15, 20]
}
forecast_data = pd.DataFrame(data)

print(forecast_data.head())
