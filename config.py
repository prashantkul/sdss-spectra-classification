#!/usr/bin/env python3
"""
Configuration file for SDSS Spectral Analysis project
"""

# Dataset configuration
DATASET_CONFIG = {
    'file_path': 'data/sdss_data.csv',
    'target_column': 'class',
    'id_columns': [
        'obj_ID', 
        'spec_obj_ID'
    ],
    'metadata_columns': [
        'run_ID', 
        'rerun_ID', 
        'cam_col', 
        'field_ID',
        'plate',
        'MJD',
        'fiber_ID'
    ],
    'photometric_columns': [
        'u', 'g', 'r', 'i', 'z'
    ],
    'spatial_columns': [
        'alpha', 'delta'
    ],
    'redshift_column': 'redshift'
}

# Model configuration
MODEL_CONFIG = {
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# Feature configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
        'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
        'redshift', 'plate', 'MJD', 'fiber_ID'
    ],
    'categorical_features': [],
    'target_classes': ['star', 'galaxy', 'quasar']
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figure_size': (10, 6),
    'dpi': 100,
    'color_palette': ['blue', 'green', 'red']
}