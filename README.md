---
title: Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 Tourism Package Prediction

## Overview
This application predicts whether a customer will purchase the **Tourism Package** based on their demographic information, travel preferences, and sales interaction data.

## Features
- **Real-time Predictions**: Get instant predictions on customer purchase likelihood
- **Probability Scores**: View confidence levels for each prediction
- **Comprehensive Input**: Analyze multiple customer attributes including:
  - Demographics (Age, Gender, Income, Occupation)
  - Travel preferences (Number of trips, Preferred hotel rating)
  - Sales interaction quality (Pitch satisfaction, Follow-ups)

## Model Information
The prediction model is trained using machine learning techniques to analyze historical customer data and identify patterns that indicate purchase likelihood.

### Key Metrics
- Customer demographics analysis
- Travel behavior patterns
- Sales pitch effectiveness
- Follow-up engagement levels

## Usage
1. Enter customer details in the input fields
2. Click "Predict Purchase Likelihood"
3. View the prediction result with confidence scores

## Technology Stack
- **Framework**: Gradio 4.0
- **ML Library**: scikit-learn
- **Data Processing**: pandas
- **Model Hosting**: HuggingFace Hub

## Model Repository
The trained model and preprocessor are hosted at:
`dararaje/Tourism_Package_Prediction`

## About
This project is part of an MLOps pipeline demonstrating:
- Automated model training and deployment
- CI/CD integration with GitHub Actions
- Model versioning and tracking with MLflow
- Production deployment on HuggingFace Spaces

---
