import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from CSV
csv_path = 'placement.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)

# Extract X and Y values from the dataset
X = df['cgpa'].values
Y = df['package'].values

# Calculate the mean of X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Calculate the slope (m) and y-intercept (b) using the formulas
numerator = np.sum((X - mean_X) * (Y - mean_Y))
denominator = np.sum((X - mean_X) ** 2)
m = numerator / denominator
b = mean_Y - m * mean_X

# Make predictions using the model
Y_pred = m * X + b

# Plot the dataset and the linear regression line
plt.scatter(X, Y, label='Actual data')
plt.plot(X, Y_pred, color='red', label='Linear Regression')
plt.xlabel('cgpa')
plt.ylabel('package')
plt.title('placement review')
plt.legend()
plt.show()
print(df.columns)