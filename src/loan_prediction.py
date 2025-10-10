
"""
Loan Approval Prediction using Machine Learning
===============================================

This module implements a complete machine learning pipeline for predicting loan approvals
based on applicant information and credit history.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class LoanApprovalPredictor:
    """
    A class for predicting loan approvals using machine learning algorithms.
    """
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load the loan prediction dataset."""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"File {filepath} not found. Please ensure the data file exists.")
            return None
    
    def explore_data(self):
        """Perform basic exploratory data analysis."""
        print("\n=== Dataset Overview ===")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nDataset info:")
        print(self.data.info())
        
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        print("\nBasic statistics:")
        print(self.data.describe())
        
        print("\nLoan Status distribution:")
        print(self.data['Loan_Status'].value_counts())
        
    def visualize_data(self):
        """Create visualizations for better understanding of the data."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Loan Approval Dataset - Exploratory Data Analysis', fontsize=16)
        
        # Loan Status distribution
        self.data['Loan_Status'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Loan Status Distribution')
        axes[0,0].set_xlabel('Loan Status')
        axes[0,0].set_ylabel('Count')
        
        # Gender distribution
        self.data['Gender'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
        axes[0,1].set_title('Gender Distribution')
        
        # Education vs Loan Status
        pd.crosstab(self.data['Education'], self.data['Loan_Status']).plot(kind='bar', ax=axes[0,2])
        axes[0,2].set_title('Education vs Loan Status')
        axes[0,2].set_xlabel('Education')
        axes[0,2].legend(['Not Approved', 'Approved'])
        
        # Property Area distribution
        self.data['Property_Area'].value_counts().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Property Area Distribution')
        axes[1,0].set_xlabel('Property Area')
        
        # Applicant Income distribution
        self.data['ApplicantIncome'].hist(bins=30, ax=axes[1,1])
        axes[1,1].set_title('Applicant Income Distribution')
        axes[1,1].set_xlabel('Income')
        
        # Credit History vs Loan Status
        pd.crosstab(self.data['Credit_History'], self.data['Loan_Status']).plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Credit History vs Loan Status')
        axes[1,2].set_xlabel('Credit History')
        axes[1,2].legend(['Not Approved', 'Approved'])
        
        plt.tight_layout()
        plt.savefig('../docs/eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        # Create a copy for processing
        df = self.data.copy()
        
        # Handle missing values
        print("\n=== Data Preprocessing ===")
        print("Handling missing values...")
        
        # Fill missing values based on domain knowledge
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        
        # Feature engineering
        print("Creating new features...")
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['Income_per_Dependent'] = df['Total_Income'] / (df['Dependents'].astype(str).str.replace('3+', '3').astype(int) + 1)
        df['Loan_Amount_per_Income'] = df['LoanAmount'] / df['Total_Income']
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                             'Self_Employed', 'Property_Area']
        
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.label_encoders[column] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
        self.label_encoders['Loan_Status'] = le_target
        
        # Select features and target
        feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                          'Credit_History', 'Property_Area', 'Total_Income', 'Income_per_Dependent',
                          'Loan_Amount_per_Income']
        
        X = df[feature_columns]
        y = df['Loan_Status']
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and compare their performance."""
        print("\n=== Model Training ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for later use
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        model_scores = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            cv_mean = cv_scores.mean()
            
            model_scores[name] = {
                'accuracy': accuracy,
                'cv_score': cv_mean,
                'predictions': y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_mean:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Select best model based on cross-validation score
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['cv_score'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name} with CV Score: {model_scores[best_model_name]['cv_score']:.4f}")
        
        return model_scores
    
    def evaluate_best_model(self):
        """Evaluate the best model in detail."""
        print(f"\n=== Detailed Evaluation of {self.best_model_name} ===")
        
        # Get predictions
        if self.best_model_name == 'Logistic Regression':
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        else:
            # For tree-based models, we need to use unscaled data
            X_test_unscaled = self.scaler.inverse_transform(self.X_test)
            y_pred = self.best_model.predict(X_test_unscaled)
            y_pred_proba = self.best_model.predict_proba(X_test_unscaled)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(12, 5))
        
        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Approved', 'Approved'],
                   yticklabels=['Not Approved', 'Approved'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(1, 2, 2)
            feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                           'Credit_History', 'Property_Area', 'Total_Income', 'Income_per_Dependent',
                           'Loan_Amount_per_Income']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title(f'Top 10 Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('../docs/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_loan_approval(self, applicant_data):
        """
        Predict loan approval for a new applicant.
        
        Parameters:
        applicant_data (dict): Dictionary containing applicant information
        
        Returns:
        prediction (str): 'Approved' or 'Not Approved'
        probability (float): Probability of approval
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([applicant_data])
        
        # Feature engineering (same as training)
        input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
        input_df['Income_per_Dependent'] = input_df['Total_Income'] / (
            input_df['Dependents'].astype(str).str.replace('3+', '3').astype(int) + 1)
        input_df['Loan_Amount_per_Income'] = input_df['LoanAmount'] / input_df['Total_Income']
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                             'Self_Employed', 'Property_Area']
        
        for column in categorical_columns:
            if column in self.label_encoders:
                input_df[column] = self.label_encoders[column].transform(input_df[column])
        
        # Select features
        feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                          'Credit_History', 'Property_Area', 'Total_Income', 'Income_per_Dependent',
                          'Loan_Amount_per_Income']
        
        X_new = input_df[feature_columns]
        
        # Scale if needed
        if self.best_model_name == 'Logistic Regression':
            X_new_scaled = self.scaler.transform(X_new)
            prediction = self.best_model.predict(X_new_scaled)[0]
            probability = self.best_model.predict_proba(X_new_scaled)[0][1]
        else:
            prediction = self.best_model.predict(X_new)[0]
            probability = self.best_model.predict_proba(X_new)[0][1]
        
        # Decode prediction
        prediction_label = self.label_encoders['Loan_Status'].inverse_transform([prediction])[0]
        
        return prediction_label, probability

def main():
    """Main function to run the loan approval prediction pipeline."""
    print("=== Loan Approval Prediction System ===")
    
    # Initialize the predictor
    predictor = LoanApprovalPredictor()
    
    # Load data
    data_path = '../data/loan_prediction.csv'
    if predictor.load_data(data_path) is None:
        print("Please ensure the dataset is available in the data folder.")
        return
    
    # Explore data
    predictor.explore_data()
    
    # Visualize data
    predictor.visualize_data()
    
    # Preprocess data
    X, y = predictor.preprocess_data()
    
    # Train models
    model_scores = predictor.train_models(X, y)
    
    # Evaluate best model
    predictor.evaluate_best_model()
    
    # Example prediction
    print("\n=== Example Prediction ===")
    sample_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    prediction, probability = predictor.predict_loan_approval(sample_applicant)
    print(f"Prediction: {prediction}")
    print(f"Approval Probability: {probability:.2%}")

if __name__ == "__main__":
    main()
