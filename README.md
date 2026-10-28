# Customer Churn Prediction API

A machine learning API that predicts whether a customer is likely to leave a service. Built with Python, FastAPI, and scikit-learn, and containerized using Docker.

## Features

- Predict customer churn using a trained ML model.
- REST API built with FastAPI.
- Input validation using Pydantic.
- Automated API tests using pytest.
- Docker containerization.
- GitHub Actions CI workflow for automated testing and Docker builds.

## Tech Stack

- Python
- FastAPI
- scikit-learn
- Pandas and NumPy
- pytest
- Docker
- GitHub Actions

## Project Structure

```text
CA-2/
├── app/
│   └── main.py
├── models/
│   └── model.joblib
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| GET | `/health` | Check application and model status |
| POST | `/predict` | Predict customer churn |

## Run Tests

```bash
python -m pytest -v
```

## Run with Docker

Build the image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run -d --name churn-ml-api -p 8000:8000 customer-churn-api
```

Access the API documentation at http://127.0.0.1:8000/docs.

## CI Pipeline

GitHub Actions automatically runs the test suite and builds the Docker image when code is pushed to `main` or a pull request targets `main`.

## Model

The project uses a trained Logistic Regression pipeline to predict customer churn from customer account information.
