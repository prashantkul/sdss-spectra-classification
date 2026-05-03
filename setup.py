#!/usr/bin/env python3
"""
Setup script for SDSS Spectral Analysis project
"""

import os
import sys
import subprocess

def check_and_install_packages():
    """Check if required packages are installed and install if missing"""
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    print("Checking required packages...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"✗ {package} is not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    
    return True

def create_project_structure():
    """Create basic project structure"""
    directories = ['data', 'models', 'results', 'notebooks']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    print("SDSS Spectral Analysis - Project Setup")
    print("=" * 40)
    
    # Check and install packages
    if not check_and_install_packages():
        print("Failed to install required packages")
        sys.exit(1)
    
    # Create project structure
    create_project_structure()
    
    print("\nSetup complete!")
    print("Project structure created:")
    print("- data/          # For raw and processed data")
    print("- models/        # For trained models")
    print("- results/       # For analysis results")
    print("- notebooks/     # For exploratory analysis")
    print("- main.py        # Main classification script")
    print("- load_data.py   # Data loading utilities")
    print("- analysis.py    # Comprehensive analysis script")
    print("- requirements.txt # Dependencies")

if __name__ == "__main__":
    main()