import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Cross border 1.csv")

# select numeric columns
numeric_df = df[['Amount','Is_laundering']]

corr = numeric_df.corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()