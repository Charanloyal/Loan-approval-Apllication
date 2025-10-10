"""
Visualization utilities for loan approval prediction project.

This module contains functions for creating various plots and visualizations
for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LoanVisualization:
    """Class containing all visualization methods for loan approval prediction."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_data_overview(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive overview plots of the dataset."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Loan Approval Dataset - Data Overview', fontsize=20, y=0.98)
        
        # Loan Status distribution
        loan_counts = df['Loan_Status'].value_counts()
        axes[0,0].pie(loan_counts.values, labels=loan_counts.index, autopct='%1.1f%%',
                     colors=self.colors[:2], startangle=90)
        axes[0,0].set_title('Loan Status Distribution', fontsize=14)
        
        # Gender distribution
        if 'Gender' in df.columns:
            df['Gender'].value_counts().plot(kind='bar', ax=axes[0,1], color=self.colors[2])
            axes[0,1].set_title('Gender Distribution', fontsize=14)
            axes[0,1].tick_params(axis='x', rotation=0)
        
        # Education vs Loan Status
        if 'Education' in df.columns:
            pd.crosstab(df['Education'], df['Loan_Status']).plot(kind='bar', ax=axes[0,2],
                       color=self.colors[:2])
            axes[0,2].set_title('Education vs Loan Status', fontsize=14)
            axes[0,2].legend(['Not Approved', 'Approved'])
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # Property Area distribution
        if 'Property_Area' in df.columns:
            df['Property_Area'].value_counts().plot(kind='bar', ax=axes[1,0], 
                                                   color=self.colors[3:6])
            axes[1,0].set_title('Property Area Distribution', fontsize=14)
            axes[1,0].tick_params(axis='x', rotation=0)
        
        # Applicant Income distribution
        if 'ApplicantIncome' in df.columns:
            axes[1,1].hist(df['ApplicantIncome'], bins=30, alpha=0.7, color=self.colors[0])
            axes[1,1].set_title('Applicant Income Distribution', fontsize=14)
            axes[1,1].set_xlabel('Income')
            axes[1,1].set_ylabel('Frequency')
        
        # Credit History vs Loan Status
        if 'Credit_History' in df.columns:
            pd.crosstab(df['Credit_History'], df['Loan_Status']).plot(kind='bar', ax=axes[1,2],
                       color=self.colors[:2])
            axes[1,2].set_title('Credit History vs Loan Status', fontsize=14)
            axes[1,2].legend(['Not Approved', 'Approved'])
            axes[1,2].tick_params(axis='x', rotation=0)
        
        # Married vs Loan Status
        if 'Married' in df.columns:
            pd.crosstab(df['Married'], df['Loan_Status']).plot(kind='bar', ax=axes[2,0],
                       color=self.colors[:2])
            axes[2,0].set_title('Marital Status vs Loan Status', fontsize=14)
            axes[2,0].legend(['Not Approved', 'Approved'])
            axes[2,0].tick_params(axis='x', rotation=0)
        
        # Loan Amount distribution
        if 'LoanAmount' in df.columns:
            axes[2,1].hist(df['LoanAmount'].dropna(), bins=30, alpha=0.7, color=self.colors[1])
            axes[2,1].set_title('Loan Amount Distribution', fontsize=14)
            axes[2,1].set_xlabel('Loan Amount')
            axes[2,1].set_ylabel('Frequency')
        
        # Self Employed vs Loan Status
        if 'Self_Employed' in df.columns:
            pd.crosstab(df['Self_Employed'], df['Loan_Status']).plot(kind='bar', ax=axes[2,2],
                       color=self.colors[:2])
            axes[2,2].set_title('Self Employment vs Loan Status', fontsize=14)
            axes[2,2].legend(['Not Approved', 'Approved'])
            axes[2,2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create correlation heatmap for numerical variables."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis.")
            return
        
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_boxplots_by_target(self, df: pd.DataFrame, target_col: str = 'Loan_Status',
                               save_path: Optional[str] = None):
        """Create box plots for numerical variables by target variable."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by {target_col}')
        
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_values: np.ndarray,
                               title: str = "Feature Importance", save_path: Optional[str] = None):
        """Plot feature importance from tree-based models."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=self.figsize)
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', 
                   palette='viridis')
        plt.title(f'{title} - Top 15 Features', fontsize=16)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        # Add value labels on bars
        for i, v in enumerate(importance_df.head(15)['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, model_results: Dict, save_path: Optional[str] = None):
        """Compare performance of multiple models."""
        models = list(model_results.keys())
        metrics = ['accuracy', 'cv_score']
        if 'auc' in model_results[models[0]]:
            metrics.append('auc')
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'Model Comparison - {metric.upper()}', fontsize=14)
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, model_results: Dict, y_test: np.ndarray,
                               save_path: Optional[str] = None):
        """Plot confusion matrices for all models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(model_results.items()):
            cm = confusion_matrix(y_test, results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Not Approved', 'Approved'],
                       yticklabels=['Not Approved', 'Approved'])
            axes[i].set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, model_results: Dict, y_test: np.ndarray,
                       save_path: Optional[str] = None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=self.figsize)
        
        for i, (name, results) in enumerate(model_results.items()):
            if 'probabilities' in results:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})',
                        color=self.colors[i % len(self.colors)], linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, model_results: Dict, y_test: np.ndarray,
                                    save_path: Optional[str] = None):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=self.figsize)
        
        for i, (name, results) in enumerate(model_results.items()):
            if 'probabilities' in results:
                precision, recall, _ = precision_recall_curve(y_test, results['probabilities'])
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})',
                        color=self.colors[i % len(self.colors)], linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create an interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loan Status Distribution', 'Income vs Loan Amount', 
                          'Credit History Impact', 'Property Area Analysis'],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Loan Status Pie Chart
        loan_counts = df['Loan_Status'].value_counts()
        fig.add_trace(
            go.Pie(labels=loan_counts.index, values=loan_counts.values,
                   name="Loan Status"),
            row=1, col=1
        )
        
        # Income vs Loan Amount Scatter
        if 'ApplicantIncome' in df.columns and 'LoanAmount' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['ApplicantIncome'], y=df['LoanAmount'],
                          mode='markers', name='Income vs Loan Amount',
                          marker=dict(color=df['Loan_Status'].map({'Y': 'green', 'N': 'red'})),
                          text=df['Loan_Status']),
                row=1, col=2
            )
        
        # Credit History Impact
        if 'Credit_History' in df.columns:
            credit_cross = pd.crosstab(df['Credit_History'], df['Loan_Status'])
            fig.add_trace(
                go.Bar(x=credit_cross.index, y=credit_cross['Y'], name='Approved'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=credit_cross.index, y=credit_cross['N'], name='Rejected'),
                row=2, col=1
            )
        
        # Property Area Analysis
        if 'Property_Area' in df.columns:
            area_cross = pd.crosstab(df['Property_Area'], df['Loan_Status'])
            fig.add_trace(
                go.Bar(x=area_cross.index, y=area_cross['Y'], name='Approved', showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=area_cross.index, y=area_cross['N'], name='Rejected', showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Loan Approval Analysis Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_missing_values(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize missing values in the dataset."""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if missing_data.empty:
            print("No missing values found in the dataset.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        missing_data.plot(kind='bar', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values Count')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Missing Values Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Percentage plot
        missing_percent = (missing_data / len(df)) * 100
        missing_percent.plot(kind='bar', ax=axes[1], color='lightblue')
        axes[1].set_title('Missing Values Percentage')
        axes[1].set_xlabel('Columns')
        axes[1].set_ylabel('Missing Values (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_comprehensive_report(df: pd.DataFrame, model_results: Dict, y_test: np.ndarray,
                              output_dir: str = "../docs/images/"):
    """Create a comprehensive visualization report."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    viz = LoanVisualization()
    
    print("Generating comprehensive visualization report...")
    
    # Data overview
    print("1. Creating data overview plots...")
    viz.plot_data_overview(df, save_path=f"{output_dir}data_overview.png")
    
    # Correlation heatmap
    print("2. Creating correlation heatmap...")
    viz.plot_correlation_heatmap(df, save_path=f"{output_dir}correlation_heatmap.png")
    
    # Box plots
    print("3. Creating box plots...")
    viz.plot_boxplots_by_target(df, save_path=f"{output_dir}boxplots_by_target.png")
    
    # Model comparison
    print("4. Creating model comparison plots...")
    viz.plot_model_comparison(model_results, save_path=f"{output_dir}model_comparison.png")
    
    # Confusion matrices
    print("5. Creating confusion matrices...")
    viz.plot_confusion_matrices(model_results, y_test, save_path=f"{output_dir}confusion_matrices.png")
    
    # ROC curves
    print("6. Creating ROC curves...")
    viz.plot_roc_curves(model_results, y_test, save_path=f"{output_dir}roc_curves.png")
    
    # Precision-Recall curves
    print("7. Creating Precision-Recall curves...")
    viz.plot_precision_recall_curves(model_results, y_test, save_path=f"{output_dir}pr_curves.png")
    
    # Missing values visualization
    print("8. Creating missing values visualization...")
    viz.plot_missing_values(df, save_path=f"{output_dir}missing_values.png")
    
    # Interactive dashboard
    print("9. Creating interactive dashboard...")
    viz.create_interactive_dashboard(df, save_path=f"{output_dir}dashboard.html")
    
    print(f"Comprehensive visualization report saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from data_utils import load_loan_data
    
    # Load sample data
    df = load_loan_data("../data/sample_data.csv")
    if df is not None:
        viz = LoanVisualization()
        viz.plot_data_overview(df)
        viz.plot_correlation_heatmap(df)
        viz.plot_missing_values(df)
