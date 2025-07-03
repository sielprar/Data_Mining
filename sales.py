import pandas as pd

# Load dataset
df = pd.read_csv("Warehouse_and_Retail_Sales.csv")

# Drop missing values
df = df.dropna()

# Drop string columns that shouldn't be encoded for modeling
df_model = df.drop(columns=['ITEM CODE', 'ITEM DESCRIPTION'])

# Encode categorical variables
df_encoded = pd.get_dummies(df_model, columns=['SUPPLIER', 'ITEM TYPE'], drop_first=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
print(plt.show())

# Monthly Retail Sales Trend
monthly_sales = df.groupby(['YEAR', 'MONTH'])['RETAIL SALES'].sum().reset_index()
sns.lineplot(data=monthly_sales, x="MONTH", y="RETAIL SALES", hue="YEAR")
plt.title("Monthly Retail Sales Trend")
print(plt.show())
