import joblib
import pandas as pd

# Load the saved model from the file
loaded_model = joblib.load('model.joblib')

# Define new data for predictions
new_data = pd.DataFrame({
    'feature1': [0, 1, 0],
    'feature2': [1, 0, 1]
})

# Use the loaded model to make predictions on the new data
predictions = loaded_model.predict(new_data)

# Print the predictions
print('Predictions for new data:')
print(predictions)
