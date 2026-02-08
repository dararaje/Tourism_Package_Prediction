---
title: Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 Tourism Package Prediction

## Overview
This application predicts whether a customer will purchase the **Tourism Package Prediction** based on their demographic information, travel preferences, and sales interaction data.

## Features
- **Interactive UI**: Beautiful Streamlit interface with organized sections
- **Real-time Predictions**: Get instant predictions on customer purchase likelihood
- **Probability Scores**: View detailed confidence levels and probability breakdowns
- **Visual Analytics**: Progress bars and metrics for easy interpretation
- **Smart Recommendations**: Actionable insights based on prediction results
- **Comprehensive Input**: Analyze multiple customer attributes including:
  - Demographics (Age, Gender, Income, Occupation)
  - Travel preferences (Number of trips, Preferred hotel rating)
  - Sales interaction quality (Pitch satisfaction, Follow-ups)

## Model Information
The prediction model is trained using machine learning techniques to analyze historical customer data and identify patterns that indicate purchase likelihood.

### Key Features Analyzed
- **Customer Demographics**: Age, gender, marital status, occupation, designation, income
- **Location & Lifestyle**: City tier, car ownership, passport status
- **Travel Behavior**: Number of trips, persons visiting, children visiting, preferred property rating
- **Sales Engagement**: Contact type, product pitched, pitch satisfaction, follow-ups, pitch duration

### Prediction Output
- **Binary Classification**: Will Purchase / Will Not Purchase
- **Confidence Score**: Percentage confidence in the prediction
- **Probability Breakdown**: Individual probabilities for both classes
- **Actionable Recommendations**: Priority levels for lead management

## Usage

### In HuggingFace Spaces
1. Enter customer details in the organized form sections:
   - **Customer Demographics** (left column)
   - **Travel Preferences** (middle column)
   - **Sales Interaction Details** (right column)
2. Click "🔮 Predict Purchase Likelihood" button
3. View the prediction result with:
   - Clear outcome (Will Purchase / Will Not Purchase)
   - Confidence percentage
   - Probability breakdown with metrics
   - Visual progress bars
   - Smart recommendations for lead prioritization

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Technology Stack
- **Framework**: Streamlit 1.28.0
- **ML Library**: scikit-learn
- **Data Processing**: pandas
- **Model Hosting**: HuggingFace Hub
- **Model Format**: joblib/pickle

## Model Repository
The trained model and preprocessor are hosted at:
`dararaje/Tourism_Package_Prediction`

Files used:
- `tourism_model.pkl` - Trained classifier model
- `tourism_preprocessor.pkl` - Feature preprocessing pipeline

## Interface Features

### Sidebar Information
- App description and purpose
- Model information summary
- Step-by-step usage guide

### Input Sections
1. **Customer Demographics**: Personal and professional details
2. **Travel Preferences**: Travel habits and preferences
3. **Sales Interaction**: Engagement and pitch details

### Output Display
- Color-coded prediction boxes (green for purchase, red for no purchase)
- Confidence metrics with delta indicators
- Visual probability distribution
- Context-aware recommendations

## Recommendations Logic

The app provides intelligent lead prioritization:

- **High Priority Lead** (>75% purchase probability): Immediate follow-up recommended
- **Moderate Priority Lead** (50-75% purchase probability): Personalized engagement suggested
- **Low Priority Lead** (>75% no purchase probability): Consider alternative approaches
- **Uncertain Lead** (45-55% range): Additional information needed

## About This Project
This application is part of an MLOps pipeline demonstrating:
- Automated model training and deployment
- CI/CD integration with GitHub Actions
- Model versioning and tracking with MLflow
- Production deployment on HuggingFace Spaces
- Interactive web interface for business users

## MLOps Pipeline Components
1. **Data Registration**: Automated dataset versioning on HuggingFace
2. **Model Training**: Systematic training with hyperparameter tuning
3. **Experiment Tracking**: MLflow for metrics and model management
4. **Model Deployment**: Automated deployment to HuggingFace Spaces
5. **CI/CD**: GitHub Actions for continuous integration

## Performance Metrics
The model performance can be tracked through:
- Training accuracy and validation metrics
- Confusion matrix analysis
- ROC-AUC scores
- Precision, recall, and F1 scores

## Future Enhancements
- A/B testing capabilities
- Model retraining pipeline
- Feature importance visualization
- Batch prediction upload
- Historical prediction tracking
- Customer feedback integration

## Support
For issues, questions, or feedback:
- Check the implementation guide in the repository
- Review HuggingFace Space logs for deployment issues
- Ensure model files are properly uploaded to the model repository

---
