import numpy as np

class TreeNode:

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeReg():

    # Initilization
    def __init__(self,num_features=None,min_sample_split=4,max_depth=None):

        self.num_features = num_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.root = None

    # 
    def fit(self,X,y):

        self.num_features = X.shape[1] if self.num_features is None else min(self.num_features,X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth = 0):

        num_samples,num_feat  = X.shape

        # Stopping Criteria 
        if (self.max_depth is not None and depth >= self.max_depth) or (num_samples <= self.min_sample_split):
            return TreeNode(value=self._avg_val(y))
        
        # Best Split
        feat_idxs = np.random.choice(num_feat,self.num_features,replace=False)
        feature_idx,threshold = self._best_split(X,y,feat_idxs)
        
        if feature_idx is None:
            return TreeNode(value=self._avg_val(y))

        # left and right
        left_idx,right_idx = self._split(X[:, feature_idx],threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return TreeNode(value=self._avg_val(y))

        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return TreeNode(feature_idx,threshold,left,right)

    # For decision tree regressor the predictions in the leaf node are just the average values.
    def _avg_val(self,y):
        avg = np.mean(y)
        return avg
    
    def _best_split(self,X,y,feat_idxs):

        best_drop = -1 
        split_feature_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:

                # information gain
                drop = self._var_drop(X_column,y,thr)

                if drop > best_drop:
                    best_drop = drop
                    split_feature_idx = feat_idx
                    split_threshold = thr

        return split_feature_idx, split_threshold
    
    def _var_drop(self,X_column,y,thr):

        # Parent var 
        parent_var = np.sum((y-np.mean(y))**2)

        # children 
        left_idx,right_idx = self._split(X_column,thr)

        # split
        y_left = y[left_idx]
        y_right = y[right_idx]

        children_var = (len(y_left)/len(y))*self._var(y_left) + (len(y_right)/len(y))*self._var(y_right)

        return parent_var-children_var

    def _split(self,X_column,thr):

        left_idx = np.argwhere(X_column<thr).flatten()
        right_idx = np.argwhere(X_column>=thr).flatten()
        return left_idx,right_idx
    
    def _var(self,y):
        var = np.sum((y-np.mean(y))**2)
        return var

    # Predictions 
    def predict(self,X):
        predictions = [self._traverse_tree(x,self.root) for x in X]
        return np.array(predictions)
    
    def _traverse_tree(self,x,node):

        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)