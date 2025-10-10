#!/usr/bin/env python3
"""
Quick test script to verify that the loan approval prediction project works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    from src.data_utils import load_loan_data
    
    data_path = 'data/sample_data.csv'
    df = load_loan_data(data_path)
    
    if df is not None:
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return True
    else:
        print("❌ Failed to load data")
        return False

def test_preprocessing():
    """Test data preprocessing functionality."""
    print("\nTesting data preprocessing...")
    from src.data_utils import load_and_preprocess_data
    
    data_path = 'data/sample_data.csv'
    processed_data, summary = load_and_preprocess_data(data_path, save_processed=False)
    
    if processed_data is not None:
        print(f"✅ Data preprocessing successful. New shape: {processed_data.shape}")
        return True
    else:
        print("❌ Data preprocessing failed")
        return False

def test_model_class():
    """Test the LoanApprovalPredictor class."""
    print("\nTesting LoanApprovalPredictor class...")
    try:
        from src.loan_prediction import LoanApprovalPredictor
        
        # Initialize predictor
        predictor = LoanApprovalPredictor()
        
        # Load data
        data_path = 'data/sample_data.csv'
        data = predictor.load_data(data_path)
        
        if data is not None:
            print("✅ LoanApprovalPredictor class works correctly")
            return True
        else:
            print("❌ LoanApprovalPredictor class failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LoanApprovalPredictor: {str(e)}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    try:
        from src.visualization import LoanVisualization
        from src.data_utils import load_loan_data
        
        # Load data
        df = load_loan_data('data/sample_data.csv')
        if df is not None:
            viz = LoanVisualization()
            print("✅ Visualization class initialized successfully")
            return True
        else:
            print("❌ Visualization test failed - no data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visualization: {str(e)}")
        return False

def test_imports():
    """Test all imports work correctly."""
    print("\nTesting package imports...")
    try:
        from src import LoanApprovalPredictor, load_loan_data, LoanVisualization
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting loan approval prediction project tests...\n")
    
    tests = [
        test_imports,
        test_data_loading,
        test_preprocessing,
        test_model_class,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\n🔗 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Add your dataset to data/loan_prediction.csv")
        print("3. Run the main script: python src/loan_prediction.py")
        print("4. Or explore the Jupyter notebook: jupyter notebook notebooks/loan_approval_eda.ipynb")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
