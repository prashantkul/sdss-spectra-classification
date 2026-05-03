#!/usr/bin/env python3
"""
SDSS Spectral Analysis - Classification Model
This script implements a machine learning model to classify astronomical objects
as stars, galaxies, or quasars based on SDSS photometric data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Load and preprocess the SDSS dataset
    In a real scenario, this would load from a CSV file
    """
    # For demonstration, we'll create sample data with the described structure
    np.random.seed(42)
    n_samples = 10000  # Reduced for demo purposes
    
    # Create sample data with the described features
    data = {
        'obj_ID': range(n_samples),
        'alpha': np.random.uniform(0, 360, n_samples),
        'delta': np.random.uniform(-90, 90, n_samples),
        'u': np.random.uniform(10, 25, n_samples),
        'g': np.random.uniform(10, 25, n_samples),
        'r': np.random.uniform(10, 25, n_samples),
        'i': np.random.uniform(10, 25, n_samples),
        'z': np.random.uniform(10, 25, n_samples),
        'run_ID': np.random.randint(1, 1000, n_samples),
        'rerun_ID': np.random.randint(1, 10, n_samples),
        'cam_col': np.random.randint(1, 7, n_samples),
        'field_ID': np.random.randint(1, 10000, n_samples),
        'spec_obj_ID': np.random.randint(1, 100000, n_samples),
        'redshift': np.random.uniform(0, 5, n_samples),
        'plate': np.random.randint(1, 10000, n_samples),
        'MJD': np.random.randint(50000, 60000, n_samples),
        'fiber_ID': np.random.randint(1, 1000, n_samples)
    }
    
    # Create class labels (star, galaxy, quasar)
    classes = ['star', 'galaxy', 'quasar']
    data['class'] = np.random.choice(classes, n_samples)
    
    df = pd.DataFrame(data)
    return df

def explore_data(df):
    """Explore the dataset structure and characteristics"""
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nClass Distribution:")
    print(df['class'].value_counts())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())

def prepare_features(df):
    """Prepare features for modeling"""
    # Select features (excluding ID columns that don't contribute to classification)
    feature_columns = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 
                      'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 
                      'redshift', 'plate', 'MJD', 'fiber_ID']
    
    X = df[feature_columns]
    y = df['class']
    
    return X, y

def train_model(X, y):
    """Train a classification model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test_scaled, y_test, y_pred

def evaluate_model(model, X_test, y_test, y_pred):
    """Evaluate the trained model"""
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Galaxy', 'Quasar', 'Star'],
                yticklabels=['Galaxy', 'Quasar', 'Star'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    """Main function to run the classification pipeline"""
    print("SDSS Spectral Analysis - Classification Model")
    print("=" * 50)
    
    # Load and explore data
    print("1. Loading and exploring data...")
    df = load_and_preprocess_data()
    explore_data(df)
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y = prepare_features(df)
    
    # Train model
    print("\n3. Training model...")
    model, scaler, X_test, y_test, y_pred = train_model(X, y)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    evaluate_model(model, X_test, y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance in Classification Model')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()