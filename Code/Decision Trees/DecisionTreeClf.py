import numpy as np 
from collections import Counter

# This Decision tree is using entropy for calculating information gain 
# A function for gini is present but is not used - to be incorporated later. 

# Node for a decision tree. 
class TreeNode:

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
# This is a Classification tree
class DecisionTreeCLF:

    def __init__(self, num_features = None, min_sample_split = 2, max_depth=100):

        # Initialisation
        '''
        num_features: number of features used for fitting the tree
        min_sample_split: This is the min number of samples that is needed to be present in the split
        max_depth: maximum height of the tree or times tree has been split
        '''

        # Stopping Criteries
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

        self.num_features = num_features
        self.root = None

    def fit(self, X, y):

        # X : features expected as a numpy array 1-D or 2-D.
        # y : target expected as a 1-D numpy array.

        self.num_features = X.shape[1] if self.num_features is None else min(self.num_features, X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth = 0):

        num_samples, num_feats = X.shape
        num_labels = len(np.unique(y))

        # Stopping Criteria
        if (self.max_depth is not None and depth >= self.max_depth) or (num_samples <= self.min_sample_split):
            return TreeNode(value=self._avg_val(y))
        
        # Best split
        feat_idxs = np.random.choice(num_feats, self.num_features, replace = False)
        feature_idx, threshold = self._best_split(X,y,feat_idxs)

        # Child Nodes
        split_column = X[:,feature_idx]
        left_idxs, right_idxs = self._split(split_column, threshold)

        left = self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return TreeNode(feature_idx, threshold, left, right)
    
    def _most_common(self,y):
        counter = Counter(y)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label
    
    def _best_split(self, X,y,feat_idxs):

        best_gain = -1
        split_feature_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

        for thr in thresholds:

            # information gain
            gain = self._information_gain(X_column,y,thr)

            if gain > best_gain:
                best_gain = gain
                split_feature_idx = feat_idx
                split_threshold = thr

        return split_feature_idx, split_threshold
    
    def _information_gain(self, X_column, y, threshold):

        # parent entropy
        parent_entropy = self._entropy(y)

        # children
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # children entropy 
        y_left, y_right = y[left_idx], y[right_idx]
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)

        child_entropy = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_entropy - child_entropy

    
    def _split(self, X_column, threshold):
        left_idx = np.argwhere(X_column < threshold).flatten()
        right_idx = np.argwhere(X_column >= threshold).flatten()
        return left_idx, right_idx

    def _entropy(self,y):
        hist = np.bincount(y)
        probs = hist/len(y)

        entropy = -np.sum(p*np.log2(p) for p in probs if p>0)
        return entropy
    
    # Gini criteria 
    def _gini(self,y):
        probs = np.bincount(y) / len(y)
        return 1 - np.sum(probs ** 2)

    def predict(self, X_test):

        return np.array([self._traverse_tree(x, self.root) for x in X_test])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    