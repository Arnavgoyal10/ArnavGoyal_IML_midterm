import pandas as pd
import numpy as np
from dtrees import DecisionTreeRegressor
import math
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=5, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        print("Starting Random Forest training...")
        self.trees = []
        n_total_features = X.shape[1]
        if self.n_features is None:
            self.n_features = n_total_features

        for i in range(self.n_trees):
            print(f"Training tree {i+1}/{self.n_trees}...")
            X_sample, y_sample = self.bootstrap_sample(X, y)
            feature_indices = np.random.choice(
                n_total_features, self.n_features, replace=False
            )
            X_sample_sub = X_sample[:, feature_indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.feature_indices = feature_indices
            tree.fit(X_sample_sub, y_sample)
            self.trees.append(tree)
        print("Random Forest training complete.")

    def predict(self, X):
        print("Starting prediction using Random Forest...")
        tree_preds = []
        for tree in self.trees:
            X_sub = X[:, tree.feature_indices]
            tree_preds.append(tree.predict(X_sub))
        tree_preds = np.array(tree_preds)
        avg_pred = np.mean(tree_preds, axis=0)
        print("Prediction complete.")
        return (avg_pred >= 0.5).astype(int)

    def feature_importances(self):
        feature_importances = np.zeros(X.shape[1])
        for tree in self.trees:
            feature_importances_for_tree = np.zeros(X.shape[1])
            for idx, feature_index in enumerate(tree.feature_indices):
                feature_importances_for_tree[feature_index] = 1
            feature_importances += feature_importances_for_tree
        feature_importances /= self.n_trees
        return feature_importances


def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = (
        (2 * (precision * recall) / (precision + recall))
        if (precision + recall) != 0
        else 0
    )

    return precision, recall, f1_score


print("Loading data...")
df = pd.read_csv("pipeline/final_merged_data.csv")
print("Data loaded successfully.")

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print("Normalizing features...")
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print("Normalization complete.")

print("Splitting data into training and testing sets...")
indices = np.arange(X_norm.shape[0])
np.random.shuffle(indices)
train_size = int(0.75 * X_norm.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X_norm[train_indices]
y_train = y[train_indices]
X_test = X_norm[test_indices]
y_test = y[test_indices]
print("Train/test split complete.")

n_total_features = X_norm.shape[1]
n_features = int(math.sqrt(n_total_features)) + 10
print(f"Total features: {n_total_features}, using {n_features} features per tree.")

rf = RandomForestRegressor(
    n_trees=30, max_depth=15, min_samples_split=5, n_features=n_features
)
rf.fit(X_train, y_train)

print("Predicting on the test set...")
predictions = rf.predict(X_test)

print("Evaluating model performance...")
accuracy = np.mean(predictions == y_test)
precision, recall, f1 = calculate_metrics(y_test, predictions)

TP = np.sum((y_test == 1) & (predictions == 1))
TN = np.sum((y_test == 0) & (predictions == 0))
FP = np.sum((y_test == 0) & (predictions == 1))
FN = np.sum((y_test == 1) & (predictions == 0))

confusion = (
    f"Confusion Matrix:\n"
    f"True Positives: {TP}\n"
    f"True Negatives: {TN}\n"
    f"False Positives: {FP}\n"
    f"False Negatives: {FN}"
)

evaluation_summary = (
    f"Accuracy: {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1-Score: {f1:.4f}\n"
    f"{confusion}\n"
)
print("Evaluation Metrics:")
print(evaluation_summary)

print("Saving confusion matrix as image...")
conf_matrix = np.array([[TN, FP], [FN, TP]])
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["Actual 0", "Actual 1"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/rf_confusion_matrix.png")
plt.close()
print("Confusion matrix image saved.")

feature_importances = rf.feature_importances()
important_features_indices = np.argsort(feature_importances)[::-1]
top_features = df.columns[important_features_indices]

print("Top features contributing to the model:")
for i, feature in enumerate(top_features[:10]):
    print(f"{i+1}. {feature}: {feature_importances[important_features_indices[i]]}")

print("Writing evaluation metrics to rforrest_accuracy_matrix.txt...")
with open("reports/rforrest_accuracy_matrix.txt", "w") as f:
    f.write(evaluation_summary)
    f.write("\nTop features contributing to the model:\n")
    for i, feature in enumerate(top_features[:10]):
        f.write(
            f"{i+1}. {feature}: {feature_importances[important_features_indices[i]]}\n"
        )
print("Done.")
