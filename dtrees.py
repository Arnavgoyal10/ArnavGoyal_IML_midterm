import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p0 = np.sum(y == 0) / len(y)
        p1 = np.sum(y == 1) / len(y)
        return 1 - (p0**2 + p1**2)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            majority_class = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            return {"type": "leaf", "class": majority_class}

        best_feature = None
        best_threshold = None
        best_impurity = float("inf")
        best_splits = None

        current_impurity = self._gini(y)
        if current_impurity == 0:
            return {"type": "leaf", "class": y[0]}

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                impurity_left = self._gini(y_left)
                impurity_right = self._gini(y_right)
                weighted_impurity = (
                    len(y_left) * impurity_left + len(y_right) * impurity_right
                ) / num_samples

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = {
                        "left_X": X[left_mask],
                        "left_y": y[left_mask],
                        "right_X": X[right_mask],
                        "right_y": y[right_mask],
                    }

        if best_feature is None:
            majority_class = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
            return {"type": "leaf", "class": majority_class}

        left_tree = self._build_tree(
            best_splits["left_X"], best_splits["left_y"], depth + 1
        )
        right_tree = self._build_tree(
            best_splits["right_X"], best_splits["right_y"], depth + 1
        )

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _predict_sample(self, x, tree):
        """Traverse the tree to predict a single sample."""
        if tree["type"] == "leaf":
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])
