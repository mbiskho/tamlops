import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Loading a datasets 
data = pd.read_csv("data/heart.csv ")

# Split data into features and target
X = data[['trtbps', 'chol', 'fbs', 'thalachh']]
y = data['output']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Export the trained model to a file using joblib
model_filename = 'model.joblib'
joblib.dump(model, model_filename)

print(f'Model saved to {model_filename}')
