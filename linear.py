import numpy as np
import pandas as pd


class LinearRegressionClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None

    def fit(self, X, y):
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.coefficients = np.zeros(X_.shape[1])

        for _ in range(self.n_iterations):
            predictions = X_ @ self.coefficients
            gradient = (X_.T @ (predictions - y)) / X_.shape[0]
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]
        predictions = X_ @ self.coefficients
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]
        predictions = X_ @ self.coefficients
        return np.clip(predictions, 0, 1)


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

print("Training Linear Regression classifier for classification...")
lin_reg_classifier = LinearRegressionClassifier(learning_rate=0.01, n_iterations=1000)
lin_reg_classifier.fit(X_train, y_train)

print("Predicting on the test set...")
predictions = lin_reg_classifier.predict(X_test)
probabilities = lin_reg_classifier.predict_proba(X_test)

print("Evaluating model performance...")
accuracy = np.mean(predictions == y_test)

TP = np.sum((y_test == 1) & (predictions == 1))
TN = np.sum((y_test == 0) & (predictions == 0))
FP = np.sum((y_test == 0) & (predictions == 1))
FN = np.sum((y_test == 1) & (predictions == 0))

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (
    (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
)

confusion = (
    f"Confusion Matrix:\n"
    f"True Positives: {TP}\n"
    f"True Negatives: {TN}\n"
    f"False Positives: {FP}\n"
    f"False Negatives: {FN}"
)

metrics = (
    f"Accuracy: {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1-Score: {f1_score:.4f}\n"
)

evaluation_summary = f"{metrics}{confusion}\n"

print("Evaluation Metrics:")
print(evaluation_summary)

print("Writing evaluation metrics to linear_regression_accuracy_matrix.txt...")
with open("reports/linear_regression_accuracy_matrix.txt", "w") as f:
    f.write(evaluation_summary)
print("Done.")
