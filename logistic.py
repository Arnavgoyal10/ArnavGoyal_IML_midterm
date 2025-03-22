import numpy as np
import pandas as pd


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_ = np.c_[np.ones(X.shape[0]), X]

        self.coefficients = np.zeros(X_.shape[1])

        for i in range(self.n_iterations):
            predictions = self.sigmoid(X_ @ self.coefficients)
            gradient = (X_.T @ (predictions - y)) / X_.shape[0]
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]
        predictions = self.sigmoid(X_ @ self.coefficients)
        return (predictions >= 0.5).astype(int)


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


print("Training Logistic Regression classifier...")
log_reg_classifier = LogisticRegressionClassifier(learning_rate=0.01, n_iterations=1000)
log_reg_classifier.fit(X_train, y_train)

print("Predicting on the test set...")
predictions = log_reg_classifier.predict(X_test)

print("Evaluating model performance...")
accuracy = np.mean(predictions == y_test)

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

evaluation_summary = f"Accuracy: {accuracy}\n{confusion}\n"
print("Evaluation Metrics:")
print(evaluation_summary)

print("Writing evaluation metrics to logistic_regression_accuracy_matrix.txt...")
with open("reports/logistic_regression_accuracy_matrix.txt", "w") as f:
    f.write(evaluation_summary)
print("Done.")
