import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

file_path = 'history_image.csv' 

df = pd.read_csv(file_path)

# Remove commas
df['gpu_usage'] = df['gpu_usage'].str.replace(',', '')
df['gpu_usage'] = df['gpu_usage'].astype(int)

# Change time column to seconds
# Convert 'time_column' to timedelta format
df['train_execution'] = pd.to_timedelta(df['train_execution'])
df['overall_execution'] = pd.to_timedelta(df['overall_execution'])

# Convert timedelta values to total seconds
df['train_execution'] = df['train_execution'].dt.total_seconds()
df['overall_execution'] = df['overall_execution'].dt.total_seconds()

#Check Data Variance
variance_multiple_columns = df.var()

# Define features (X) and the target variable (y)
columns_to_drop = ['train_execution', 'overall_execution', 'file', 'cpu_usage', 'gpu_usage', 'num_dataset']  # Replace with the names of columns you want to drop
X = df.drop(columns=columns_to_drop, axis=1)  # Drop the train_execution column
y = df[['overall_execution', 'gpu_usage']]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict train_execution_time for the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example feature values for prediction
new_features = pd.DataFrame({
    'resolution': [1000],
    'train_batch_size': [2],
    'num_train_epochs': [2],
    'max_train_steps': [100],
    'learning_rate': [0.0001],
    'gradient_accumulation_steps': [1],
    'file_size': [6232832]
})

# Use the trained model to predict the target variable for the new feature values
predicted_y = model.predict(new_features)

# Clip the predicted values to be non-negative
predicted_y_non_negative = np.maximum(predicted_y, 0)

print(f"Predicted Execution Time: {predicted_y_non_negative[0][0]}")
print(f"Predicted GPU Memory Usage: {predicted_y_non_negative[0][1]}")

# Save the trained model to a .pkl file
with open('text_to_image_model.pkl', 'wb') as file:
    pickle.dump(model, file)