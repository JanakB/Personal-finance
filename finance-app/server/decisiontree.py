import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Check termination criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(unique_classes) == 1):
            return {'class': self._most_common_label(y)}

        # Find best split
        best_split = self._find_best_split(X, y)

        # Split recursively
        if best_split['impurity'] > 0:
            left_idxs, right_idxs = best_split['left_idxs'], best_split['right_idxs']
            left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
            right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
            return {'feature_idx': best_split['feature_idx'],
                    'threshold': best_split['threshold'],
                    'impurity': best_split['impurity'],
                    'left': left_tree, 'right': right_tree}
        else:
            return {'class': self._most_common_label(y)}

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_split = {'impurity': float('inf')}

        # Calculate impurity before splitting
        current_impurity = self._calculate_impurity(y)

        # Loop through each feature
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                right_idxs = np.where(X[:, feature_idx] > threshold)[0]

                # Skip if split results in empty children
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate impurity after splitting
                left_impurity = self._calculate_impurity(y[left_idxs])
                right_impurity = self._calculate_impurity(y[right_idxs])
                weighted_impurity = (len(left_idxs) / n_samples) * left_impurity + \
                                    (len(right_idxs) / n_samples) * right_impurity

                # Update best split if necessary
                if weighted_impurity < best_split['impurity']:
                    best_split = {'feature_idx': feature_idx,
                                  'threshold': threshold,
                                  'impurity': weighted_impurity,
                                  'left_idxs': left_idxs,
                                  'right_idxs': right_idxs}

        return best_split

    def _calculate_impurity(self, y):
        # Gini impurity calculation
        classes = np.unique(y)
        n_samples = len(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / n_samples
            impurity -= p ** 2
        return impurity

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x, tree=None):
        if tree is None:
            tree = self.tree

        if 'class' in tree:
            return tree['class']
        else:
            feature_idx = tree['feature_idx']
            threshold = tree['threshold']
            if x[feature_idx] <= threshold:
                return self._predict(x, tree['left'])
            else:
                return self._predict(x, tree['right'])

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
