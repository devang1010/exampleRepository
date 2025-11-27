import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('/tmp/dataset02.csv')
print(f"Original Data Shape: {data.shape}")
print(f"Original Data:\n{data.head()}")

# Step - 1: Data Cleaning 
# Check data types
print("\n--- Data Types ---")
print(data.dtypes)

# Try to convert 'x' to numeric, coercing errors
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data_clean = data.dropna() # Now drop NaN values

print(f"\nCleaned Data Shape: {data_clean.shape}")
print(f"Cleaned Data:\n{data_clean.head()}")

# Step - 2: Outlier Removal using IQR (Interquartile Range) Method

# 1. Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
Q1_x = data_clean['x'].quantile(0.25)
Q3_x = data_clean['x'].quantile(0.75)
Q1_y = data_clean['y'].quantile(0.25)
Q3_y = data_clean['y'].quantile(0.75)

# 2. Calculate IQR (Interquartile Range) for each column
IQR_x = Q3_x - Q1_x
IQR_y = Q3_y - Q1_y

print(f"\n--- IQR Statistics ---")
print(f"X - Q1: {Q1_x}, Q3: {Q3_x}, IQR: {IQR_x}")
print(f"Y - Q1: {Q1_y}, Q3: {Q3_y}, IQR: {IQR_y}")

# 3. Define outlier bounds (typically 1.5 * IQR)
lower_bound_x = Q1_x - 1.5 * IQR_x
upper_bound_x = Q3_x + 1.5 * IQR_x
lower_bound_y = Q1_y - 1.5 * IQR_y
upper_bound_y = Q3_y + 1.5 * IQR_y

print(f"\n--- Outlier Bounds ---")
print(f"X bounds: [{lower_bound_x}, {upper_bound_x}]")
print(f"Y bounds: [{lower_bound_y}, {upper_bound_y}]")

# 4. Filter out outliers - keep only values within bounds for BOTH x and y
data_no_outliers = data_clean[
    (data_clean['x'] >= lower_bound_x) & (data_clean['x'] <= upper_bound_x) &
    (data_clean['y'] >= lower_bound_y) & (data_clean['y'] <= upper_bound_y)
]

removed_count = len(data_clean) - len(data_no_outliers)
print(f"\nAfter Outlier Removal Data Shape: {data_no_outliers.shape}")
print(f"Removed {removed_count} outliers using IQR method")
print(f"Data after outlier removal:\n{data_no_outliers.head()}")

# Step - 3: Data Normalization (Min-Max Scaling)
# Normalize data to range [0, 1]

# Store original min and max values for potential denormalization later
min_x = data_no_outliers['x'].min()
max_x = data_no_outliers['x'].max()
min_y = data_no_outliers['y'].min()
max_y = data_no_outliers['y'].max()

print(f"\n--- Original Data Range ---")
print(f"X range: [{min_x}, {max_x}]")
print(f"Y range: [{min_y}, {max_y}]")

# Apply Min-Max normalization: (value - min) / (max - min)
data_normalized = data_no_outliers.copy()
data_normalized['x'] = (data_no_outliers['x'] - min_x) / (max_x - min_x)
data_normalized['y'] = (data_no_outliers['y'] - min_y) / (max_y - min_y)

print(f"\n--- Normalized Data ---")
print(f"Normalized Data Shape: {data_normalized.shape}")
print(f"Normalized Data (first 5 rows):\n{data_normalized.head()}")
print(f"\nNormalized Data Statistics:")
print(data_normalized.describe())

# Step - 4: Train-Test Split (80% training, 20% testing)
# Manual split without sklearn

# Shuffle the data randomly with a fixed seed for reproducibility
np.random.seed(42)
data_shuffled = data_normalized.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the split index (80% for training)
split_index = int(0.8 * len(data_shuffled))

# Split into training and testing sets
train_data = data_shuffled[:split_index]
test_data = data_shuffled[split_index:]

print(f"\n--- Train-Test Split ---")
print(f"Training Data Shape: {train_data.shape} ({len(train_data)/len(data_normalized)*100:.1f}%)")
print(f"Testing Data Shape: {test_data.shape} ({len(test_data)/len(data_normalized)*100:.1f}%)")
print(f"\nTraining Data (first 5 rows):\n{train_data.head()}")
print(f"\nTesting Data (first 5 rows):\n{test_data.head()}")

# Save the training and testing datasets to CSV files
train_data.to_csv('/tmp/dataset02_training.csv', index=False)
test_data.to_csv('/tmp/dataset02_testing.csv', index=False)

print(f"\n--- Files Saved ---")
print(f"Training data saved to: /tmp/dataset02_training.csv")
print(f"Testing data saved to: /tmp/dataset02_testing.csv")

# Step - 5: Build OLS Model using TRAINING DATA ONLY

# Prepare training data for OLS
X_train = train_data['x']  # Independent variable (influence data)
y_train = train_data['y']  # Dependent variable (target variable)

# Add constant term (intercept) to the model
X_train_with_const = sm.add_constant(X_train)

# Fit the OLS model using ONLY training data
ols_model = sm.OLS(y_train, X_train_with_const).fit()

# Print the model summary
print(f"\n--- OLS Model Summary (Training Data Only) ---")
print(ols_model.summary())

# Extract model parameters
intercept = ols_model.params['const']
slope = ols_model.params['x']

print(f"\n--- Model Parameters ---")
print(f"Intercept (β0): {intercept:.6f}")
print(f"Slope (β1): {slope:.6f}")
print(f"R-squared: {ols_model.rsquared:.6f}")
print(f"Adjusted R-squared: {ols_model.rsquared_adj:.6f}")
print(f"F-statistic: {ols_model.fvalue:.6f}")
print(f"Prob (F-statistic): {ols_model.f_pvalue:.6e}")

# Generate predictions for BOTH training and testing data
# (We'll need these for visualization in the next steps)
X_test = test_data['x']
X_test_with_const = sm.add_constant(X_test)

y_train_pred = ols_model.predict(X_train_with_const)
y_test_pred = ols_model.predict(X_test_with_const)

print(f"\n--- Predictions Generated ---")
print(f"Training predictions shape: {y_train_pred.shape}")
print(f"Testing predictions shape: {y_test_pred.shape}")

# Step - 6: Visualization 1 - Scatter Plot with OLS Line
# Create figure for scatter plot
plt.figure(figsize=(10, 6))

# Plot training data in orange
plt.scatter(train_data['x'], train_data['y'], 
           color='orange', alpha=0.6, label='Training Data', s=50)

# Plot testing data in blue
plt.scatter(test_data['x'], test_data['y'], 
           color='blue', alpha=0.6, label='Testing Data', s=50)

# Create red line plot for OLS model
# Sort x values for smooth line plotting
X_line = np.linspace(train_data['x'].min(), train_data['x'].max(), 100)
X_line_with_const = sm.add_constant(X_line)
y_line = ols_model.predict(X_line_with_const)

plt.plot(X_line, y_line, color='red', linewidth=2, label='OLS Model')

# Add labels and title
plt.xlabel('X (Influence Variable)', fontsize=12)
plt.ylabel('Y (Target Variable)', fontsize=12)
plt.title('Scatter Plot with OLS Model Line', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save figure as PDF
plt.tight_layout()
plt.savefig('/tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf', format='pdf')
print(f"\n--- Scatter Plot Saved ---")
print(f"File saved: /tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf")

# Close the figure to free memory
plt.close()

# Step - 7: Visualization 2 - Box Plot

# Create figure for box plot
plt.figure(figsize=(10, 6))

# Create box plot for both x and y dimensions
# Combine training and testing data for complete view
all_data = pd.concat([train_data, test_data])
plt.boxplot([all_data['x'], all_data['y']], 
            labels=['X (Influence Variable)', 'Y (Target Variable)'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))

# Add labels and title
plt.ylabel('Normalized Values', fontsize=12)
plt.title('Box Plot of All Data Dimensions', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Save figure as PDF
plt.tight_layout()
plt.savefig('/tmp/UE_04_App2_BoxPlot.pdf', format='pdf')
print(f"\n--- Box Plot Saved ---")
print(f"File saved: /tmp/UE_04_App2_BoxPlot.pdf")

# Close the figure to free memory
plt.close()

# Step - 8: Visualization 3 - Diagnostic Plots using provided class
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic

print(f"\n--- Creating Diagnostic Plots using LinearRegDiagnostic ---")

# Create diagnostic plots using the provided class
cls = LinearRegDiagnostic(ols_model)

# Generate all diagnostic plots
fig = cls(plot_context='seaborn-v0_8-paper')

# Save the diagnostic plots as PDF
plt.savefig('/tmp/UE_04_App2_DiagnosticPlots.pdf', format='pdf')
print(f"\n--- Diagnostic Plots Saved ---")
print(f"File saved: /tmp/UE_04_App2_DiagnosticPlots.pdf")

# Close the figure to free memory
plt.close()

# Also display VIF table
print(f"\n--- VIF Table ---")
print(cls.vif_table())

print(f"\n=== All Visualizations Complete ===")
print(f"Three PDF files have been created:")
print(f"1. UE_04_App2_ScatterVisualizationAndOlsModel.pdf")
print(f"2. UE_04_App2_BoxPlot.pdf")
print(f"3. UE_04_App2_DiagnosticPlots.pdf")