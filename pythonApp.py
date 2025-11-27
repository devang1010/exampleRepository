import pandas as pd
import numpy as np
import statsmodels.api as sm

print(f"Hello World")

# Load the csv file 
data = pd.read_csv('/tmp/dataset01.csv')

# 1. Display the number of entries in the y column
num_entries = len(data['y'])
print(f"Number of entries in the y column: {num_entries}")

# 2. Mean of column 'y'
mean_y = data['y'].mean()
print(f"Mean of the y column: {mean_y}")

# 3. Standard deviation of column 'y'
std_y = data['y'].std()
print(f"Standard deviation of the y column: {std_y}")

# 4. Variance of column 'y'
var_y = data['y'].var()
print(f"Variance of the y column: {var_y}")

# 5. Min and Max of column 'y'
min_y = data['y'].min()
max_y = data['y'].max()
print(f"Min of the y column: {min_y}")
print(f"Max of the y column: {max_y}")

# 6. OLS Model (x predicts y) An OLS (Ordinary Least Squares) model is a statistical technique for estimating the parameters of a linear regression model
X = data['x']
Y = data['y']
X = sm.add_constant(X)  # Adds intercept term
model = sm.OLS(Y, X).fit()
print("\nOLS Model Summary:")
print(model.summary())

# Save the model
model.save('/tmp/OLS_model')
print("\nModel saved to /tmp/OLS_model")
      


