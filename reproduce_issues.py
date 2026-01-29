
import numpy as np
import sys
import os

# Add Code directories to path
sys.path.append(os.path.abspath("Code/Decision Trees"))
sys.path.append(os.path.abspath("Code/Logistic Regression"))
sys.path.append(os.path.abspath("Code/Linear Regression"))

def test_decision_tree_clf():
    print("Testing DecisionTreeClf...")
    from DecisionTreeClf import DecisionTreeCLF
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeCLF(min_sample_split=2, max_depth=1)
    try:
        clf.fit(X, y)
        print("DecisionTreeClf fit successful")
    except Exception as e:
        print(f"DecisionTreeClf fit failed: {e}")

    # Test missing _avg_val by forcing a leaf node creation via max_depth
    clf_leaf = DecisionTreeCLF(max_depth=0)
    try:
        clf_leaf.fit(X, y)
        print("DecisionTreeClf leaf creation successful")
    except Exception as e:
        print(f"DecisionTreeClf leaf creation failed: {e}")

def test_decision_tree_reg():
    print("\nTesting DecisionTreeReg...")
    from DecisionTreeReg import DecisionTreeReg
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    reg = DecisionTreeReg(min_sample_split=2)
    reg.fit(X, y)
    
    # Check if _var returns SSE or Variance
    y_test = np.array([1, 2, 3])
    # Mean = 2. SSE = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2. Variance = 2/3 = 0.66
    var_val = reg._var(y_test)
    print(f"_var([1, 2, 3]) returned: {var_val}")
    if abs(var_val - 2.0) < 1e-5:
        print("Confirmed: _var returns SSE")
    elif abs(var_val - 0.666666) < 1e-5:
        print("Confirmed: _var returns Variance")
    else:
        print("Unknown return value for _var")

def test_logistic_regression():
    print("\nTesting LogisticRegressionGD...")
    from logistic_regression_gd import LogisticRegressionGD
    
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Test with large alpha (default 0.05) and unscaled gradient
    # If unscaled, gradient will be ~100/2 * 0.5 = 25. 
    # Update = 0.05 * 25 = 1.25. This is large but maybe stable.
    # If we increase N to 1000, gradient ~ 250. Update ~ 12.5. Unstable.
    
    lr = LogisticRegressionGD(itr=10)
    lr.fit(X, y)
    print(f"Beta after 10 iterations (N=100): {lr.beta}")
    
    X_large = np.random.randn(1000, 2)
    y_large = (X_large[:, 0] + X_large[:, 1] > 0).astype(int)
    
    lr_large = LogisticRegressionGD(itr=10)
    lr_large.fit(X_large, y_large)
    print(f"Beta after 10 iterations (N=1000): {lr_large.beta}")
    
    if np.any(np.isnan(lr_large.beta)) or np.any(np.abs(lr_large.beta) > 100):
        print("Instability detected with large N due to unscaled gradient")

if __name__ == "__main__":
    test_decision_tree_clf()
    test_decision_tree_reg()
    test_logistic_regression()
