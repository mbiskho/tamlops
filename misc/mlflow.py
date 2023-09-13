# import mlflow
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.pipeline import Pipeline

# # Step 1: Load the data
# data = load_iris()
# X = data.data
# y = data.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 2: Define the scikit-learn pipeline
# pipeline = Pipeline([
#     ('classifier', RandomForestClassifier(n_estimators=100))
# ])

# # Step 3: Set up MLflow tracking
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your MLflow server URI
# mlflow.set_experiment("Default")

# # Step 3: Log the dataset information
# with mlflow.start_run():
#     mlflow.log_params({'n_estimators': 100})
    
#     # Log dataset information
#     mlflow.log_params({'dataset': 'Iris'})
#     mlflow.log_params({'dataset_shape': X.shape})
    
#     # Define a description of the dataset
#     dataset_description = "This is the Iris dataset from scikit-learn."
#     mlflow.log_params({'dataset_description': dataset_description})
    
#     # Fit the pipeline to the training data
#     pipeline = Pipeline([
#         ('classifier', RandomForestClassifier(n_estimators=100))
#     ])
#     pipeline.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = pipeline.predict(X_test)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Log metrics and the trained model
#     mlflow.log_metrics({'accuracy': accuracy})
#     mlflow.sklearn.log_model(pipeline, "model")
