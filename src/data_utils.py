"""
Data utilities for loan approval prediction project.

This module contains functions for data loading, preprocessing, and validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_loan_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load loan prediction dataset from CSV file.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pd.DataFrame or None: Loaded dataframe or None if file not found
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File {filepath} not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset has the expected schema.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    bool: True if schema is valid, False otherwise
    """
    expected_columns = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
        'Property_Area', 'Loan_Status'
    ]
    
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        return False
    
    logger.info("Data schema validation passed.")
    return True


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive summary of the dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'target_distribution': df['Loan_Status'].value_counts().to_dict() if 'Loan_Status' in df.columns else {}
    }
    
    # Categorical columns analysis
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area', 'Loan_Status']
    summary['categorical_summary'] = {}
    
    for col in categorical_cols:
        if col in df.columns:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    # Numerical columns analysis
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                     'Loan_Amount_Term', 'Credit_History']
    summary['numerical_summary'] = {}
    
    for col in numerical_cols:
        if col in df.columns:
            summary['numerical_summary'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'null_count': df[col].isnull().sum()
            }
    
    return summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset using appropriate strategies.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with missing values
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    df_processed = df.copy()
    
    logger.info("Handling missing values...")
    logger.info(f"Missing values before processing:\n{df_processed.isnull().sum()}")
    
    # Fill categorical variables with mode
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for column in categorical_columns:
        if column in df_processed.columns and df_processed[column].isnull().any():
            mode_value = df_processed[column].mode()[0] if not df_processed[column].mode().empty else 'Unknown'
            df_processed[column].fillna(mode_value, inplace=True)
            logger.info(f"Filled {column} missing values with mode: {mode_value}")
    
    # Fill numerical variables with appropriate values
    if 'LoanAmount' in df_processed.columns:
        median_loan = df_processed['LoanAmount'].median()
        df_processed['LoanAmount'].fillna(median_loan, inplace=True)
        logger.info(f"Filled LoanAmount missing values with median: {median_loan}")
    
    if 'Loan_Amount_Term' in df_processed.columns:
        mode_term = df_processed['Loan_Amount_Term'].mode()[0] if not df_processed['Loan_Amount_Term'].mode().empty else 360
        df_processed['Loan_Amount_Term'].fillna(mode_term, inplace=True)
        logger.info(f"Filled Loan_Amount_Term missing values with mode: {mode_term}")
    
    if 'Credit_History' in df_processed.columns:
        mode_credit = df_processed['Credit_History'].mode()[0] if not df_processed['Credit_History'].mode().empty else 1
        df_processed['Credit_History'].fillna(mode_credit, inplace=True)
        logger.info(f"Filled Credit_History missing values with mode: {mode_credit}")
    
    # Fill CoapplicantIncome with 0 (assuming no coapplicant)
    if 'CoapplicantIncome' in df_processed.columns:
        df_processed['CoapplicantIncome'].fillna(0, inplace=True)
        logger.info("Filled CoapplicantIncome missing values with 0")
    
    logger.info(f"Missing values after processing:\n{df_processed.isnull().sum()}")
    
    return df_processed


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features from existing data.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with new engineered features
    """
    df_engineered = df.copy()
    
    logger.info("Creating engineered features...")
    
    # Total Income
    if 'ApplicantIncome' in df_engineered.columns and 'CoapplicantIncome' in df_engineered.columns:
        df_engineered['Total_Income'] = df_engineered['ApplicantIncome'] + df_engineered['CoapplicantIncome']
        logger.info("Created Total_Income feature")
    
    # Income per Dependent
    if 'Total_Income' in df_engineered.columns and 'Dependents' in df_engineered.columns:
        dependents_numeric = df_engineered['Dependents'].astype(str).str.replace('3+', '3').astype(int)
        df_engineered['Income_per_Dependent'] = df_engineered['Total_Income'] / (dependents_numeric + 1)
        logger.info("Created Income_per_Dependent feature")
    
    # Loan Amount per Income ratio
    if 'LoanAmount' in df_engineered.columns and 'Total_Income' in df_engineered.columns:
        df_engineered['Loan_Amount_per_Income'] = df_engineered['LoanAmount'] / df_engineered['Total_Income']
        # Handle division by zero
        df_engineered['Loan_Amount_per_Income'] = df_engineered['Loan_Amount_per_Income'].replace([np.inf, -np.inf], np.nan)
        df_engineered['Loan_Amount_per_Income'].fillna(df_engineered['Loan_Amount_per_Income'].median(), inplace=True)
        logger.info("Created Loan_Amount_per_Income feature")
    
    # Log transformations for skewed features
    if 'ApplicantIncome' in df_engineered.columns:
        df_engineered['ApplicantIncome_Log'] = np.log1p(df_engineered['ApplicantIncome'])
        logger.info("Created ApplicantIncome_Log feature")
    
    if 'LoanAmount' in df_engineered.columns:
        df_engineered['LoanAmount_Log'] = np.log1p(df_engineered['LoanAmount'])
        logger.info("Created LoanAmount_Log feature")
    
    # Binary features
    if 'Credit_History' in df_engineered.columns:
        df_engineered['Has_Credit_History'] = (df_engineered['Credit_History'] == 1).astype(int)
        logger.info("Created Has_Credit_History feature")
    
    if 'CoapplicantIncome' in df_engineered.columns:
        df_engineered['Has_Coapplicant'] = (df_engineered['CoapplicantIncome'] > 0).astype(int)
        logger.info("Created Has_Coapplicant feature")
    
    logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
    
    return df_engineered


def detect_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr') -> Dict:
    """
    Detect outliers in specified columns using IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to check for outliers
    method (str): Method to use ('iqr' or 'zscore')
    
    Returns:
    dict: Dictionary containing outlier information
    """
    outlier_info = {}
    
    for column in columns:
        if column not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            outlier_info[column] = {
                'method': 'IQR',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(df) * 100,
                'outlier_indices': outliers.index.tolist()
            }
            
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            threshold = 3
            
            z_scores = np.abs((df[column] - mean) / std)
            outliers = df[z_scores > threshold]
            
            outlier_info[column] = {
                'method': 'Z-score',
                'threshold': threshold,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(df) * 100,
                'outlier_indices': outliers.index.tolist()
            }
    
    return outlier_info


def save_processed_data(df: pd.DataFrame, filepath: str) -> bool:
    """
    Save processed dataframe to CSV file.
    
    Parameters:
    df (pd.DataFrame): Processed dataframe
    filepath (str): Path to save the file
    
    Returns:
    bool: True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        return False


def load_and_preprocess_data(filepath: str, save_processed: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data loading and preprocessing pipeline.
    
    Parameters:
    filepath (str): Path to the raw data file
    save_processed (bool): Whether to save processed data
    
    Returns:
    tuple: (processed_dataframe, summary_statistics)
    """
    logger.info("Starting complete data preprocessing pipeline...")
    
    # Load data
    df = load_loan_data(filepath)
    if df is None:
        return None, None
    
    # Validate schema
    if not validate_data_schema(df):
        return None, None
    
    # Get initial summary
    initial_summary = get_data_summary(df)
    logger.info(f"Initial data summary: {initial_summary['shape']}")
    
    # Handle missing values
    df_processed = handle_missing_values(df)
    
    # Create engineered features
    df_processed = create_engineered_features(df_processed)
    
    # Detect outliers
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    outlier_info = detect_outliers(df_processed, numerical_cols)
    
    # Get final summary
    final_summary = get_data_summary(df_processed)
    final_summary['outlier_info'] = outlier_info
    
    # Save processed data if requested
    if save_processed:
        processed_filepath = filepath.replace('.csv', '_processed.csv')
        save_processed_data(df_processed, processed_filepath)
    
    logger.info("Data preprocessing pipeline completed successfully!")
    
    return df_processed, final_summary


if __name__ == "__main__":
    # Example usage
    sample_path = "../data/sample_data.csv"
    processed_data, summary = load_and_preprocess_data(sample_path, save_processed=True)
    
    if processed_data is not None:
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Final columns: {list(processed_data.columns)}")
        print("\nSample of processed data:")
        print(processed_data.head())
