# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv("cleaned_dataset.csv")

num_cols = data.select_dtypes(include=['int64','float64']).columns

minmax = MinMaxScaler()
normalized_data = data.copy()

normalized_data[num_cols] = minmax.fit_transform(data[num_cols])

print("Normalized Dataset (Selected Columns):")
print(normalized_data[num_cols].head())

normalized_data.to_csv("normalized_dataset.csv", index=False)