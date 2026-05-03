#!/usr/bin/env python3
"""
Comprehensive analysis of SDSS dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_distribution(df):
    """Analyze the distribution of classes and features"""
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total observations: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1} (excluding class)")
    print(f"Class distribution:")
    
    class_dist = df['class'].value_counts()
    for class_name, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    return class_dist

def visualize_class_distribution(df):
    """Visualize class distribution"""
    plt.figure(figsize=(10, 6))
    
    # Bar plot of class distribution
    plt.subplot(1, 2, 1)
    class_counts = df['class'].value_counts()
    bars = plt.bar(class_counts.index, class_counts.values, color=['blue', 'green', 'red'])
    plt.title('Distribution of Object Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
            colors=['blue', 'green', 'red'], startangle=90)
    plt.title('Class Distribution (Pie Chart)')
    
    plt.tight_layout()
    plt.show()

def analyze_numerical_features(df):
    """Analyze numerical features"""
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns that don't contribute to analysis
    id_cols = ['obj_ID', 'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'plate', 'fiber_ID']
    feature_cols = [col for col in numerical_cols if col not in id_cols]
    
    print(f"\n=== NUMERICAL FEATURES ANALYSIS ===")
    print(f"Numerical features: {feature_cols}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df[feature_cols].describe())
    
    return feature_cols

def visualize_feature_correlations(df, feature_cols):
    """Visualize correlations between features"""
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def visualize_feature_distributions(df, feature_cols):
    """Visualize distributions of key features"""
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        for class_name in df['class'].unique():
            data = df[df['class'] == class_name][feature]
            ax.hist(data, alpha=0.7, label=class_name, bins=30)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def perform_pca_analysis(df, feature_cols):
    """Perform PCA analysis to understand data structure"""
    # Prepare data
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, 11), pca.explained_variance_ratio_[:10])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance of First 10 Components')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== PCA ANALYSIS ===")
    print(f"Total components: {len(pca.explained_variance_ratio_)}")
    print(f"First 5 components explain {np.sum(pca.explained_variance_ratio_[:5]):.2%} of variance")
    print(f"First 10 components explain {np.sum(pca.explained_variance_ratio_[:10]):.2%} of variance")

def main():
    """Main analysis function"""
    print("SDSS Dataset Analysis")
    print("=" * 50)
    
    # For demonstration, we'll use the same sample data as in main.py
    # In practice, you would load your actual data
    np.random.seed(42)
    n_samples = 10000
    
    # Create sample data with the described structure
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
    
    # Create class labels
    classes = ['star', 'galaxy', 'quasar']
    data['class'] = np.random.choice(classes, n_samples)
    
    df = pd.DataFrame(data)
    
    # Perform analysis
    class_dist = analyze_dataset_distribution(df)
    visualize_class_distribution(df)
    
    feature_cols = analyze_numerical_features(df)
    visualize_feature_correlations(df, feature_cols)
    visualize_feature_distributions(df, feature_cols)
    perform_pca_analysis(df, feature_cols)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()