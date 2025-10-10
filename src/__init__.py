"""
Loan Approval Prediction Package

This package provides tools for predicting loan approvals using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .loan_prediction import LoanApprovalPredictor
from .data_utils import load_loan_data, handle_missing_values, create_engineered_features
from .visualization import LoanVisualization, create_comprehensive_report

__all__ = [
    'LoanApprovalPredictor',
    'load_loan_data',
    'handle_missing_values',
    'create_engineered_features',
    'LoanVisualization',
    'create_comprehensive_report'
]
