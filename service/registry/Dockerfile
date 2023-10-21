# Use a base image with Python and a package manager (e.g., Anaconda)
FROM continuumio/anaconda3:latest

# Set working directory
WORKDIR /app

# Install MLflow and other dependencies
RUN conda install -c conda-forge mlflow=1.18.0 \
    && conda clean -afy

# Expose the MLflow server port (default is 5000)
EXPOSE 5000

# Start the MLflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000
