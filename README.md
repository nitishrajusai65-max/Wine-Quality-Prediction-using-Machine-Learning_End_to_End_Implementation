End-to-End Wine Quality Prediction MLOps Pipeline
This project demonstrates a complete MLOps workflow for training a machine learning model and deploying it as a containerized web application. The goal is to predict the quality of red wine based on its chemical properties.

This implementation is based on the provided Jupyter Notebook (Wine (1).ipynb) and dataset (winequality.csv), fulfilling the requirements of the MLOps Take-Home Assessment.

Project Structure
.
├── Dockerfile              # Instructions to build the Docker container
├── index.html              # Frontend UI for interacting with the model
├── main.py                 # FastAPI application to serve the model
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── train.py                # Script to train and save the model
└── winequality.csv         # The raw dataset

What Was Implemented and Why
I chose to implement a full model-to-production pipeline, which includes:

A Training Script (train.py): The code from the original Jupyter Notebook was refactored into a standalone Python script. This is a crucial MLOps practice as it makes the training process reproducible, versionable, and easy to automate. The script handles data loading, preprocessing, training the Random Forest model (which performed best in the notebook), and serializing the trained model and data scaler to disk.

A Deployable API (main.py): The trained model is wrapped in a REST API using FastAPI. This decouples the model from a specific application and allows any client (e.g., a web browser, a mobile app, another service) to get predictions over the network. This is a standard pattern for serving ML models in production.

A Simple Frontend (index.html): To provide a user-friendly way to test the deployed model, a simple HTML page with JavaScript was created. It sends a request to the FastAPI backend and displays the returned prediction.

Containerization (Dockerfile): The entire application (FastAPI server, model artifacts, and dependencies) is packaged into a Docker container. This ensures that the application runs consistently across different environments (local machine, staging, production) and simplifies deployment.

This end-to-end approach demonstrates key MLOps principles: reproducibility, automation, and deployability.

How to Run This Project
Prerequisites
Python 3.8+

Docker

Option 1: Running Locally
Clone the repository and navigate to the project directory.

Install dependencies:

pip install -r requirements.txt

Run the training script: This will create wine_model.joblib and scaler.joblib.

python train.py

Start the API server:

uvicorn main:app --reload

Access the application: Open your web browser and go to http://127.0.0.1:8000.

Option 2: Running with Docker (Recommended)
Build the Docker image:

docker build -t wine-quality-app .

Run the Docker container:

docker run -p 8000:8000 wine-quality-app

Access the application: Open your web browser and go to http://127.0.0.1:8000.

Assumptions and Limitations
Model Selection: The Random Forest Classifier was chosen as it had the highest F1-score in the provided notebook. No further hyperparameter tuning was performed, as the focus was on the MLOps workflow.

Feature Engineering: The preprocessing steps are identical to those in the notebook. No additional feature engineering was explored.

Error Handling: The current API has basic error handling. In a production system, more robust validation and logging would be necessary.
