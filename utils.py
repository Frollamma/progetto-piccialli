from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    silhouette_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, ClassifierMixin

LABELS = ["class_target", "value_target"]

class IdentityScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return X

class RegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor_model, intervals, classes_map = None):
        self.model = regressor_model
        self.intervals = intervals

        if classes_map is None:
            self.classes_map = {i: i for i in range(len(intervals) - 1)}
        else:
            self.classes_map = classes_map

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred_class = np.zeros_like(y_pred)
        for i in range(len(self.intervals) - 1):
            # Note: mask is a vector of booleans
            mask = (y_pred >= self.intervals[i]) & (y_pred < self.intervals[i + 1])
            y_pred_class[mask] = self.classes_map[i]
        return y_pred_class

def get_features(df, labels=LABELS):
    return [col for col in df.columns if col not in LABELS]

def plot_confusion_matrix(y_true, y_pred, classes=None):
    # Create the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    if classes is None:
        classes = np.unique(y_true)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="coolwarm", fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    tick_marks = np.arange(len(classes)) + 0.5

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.show()


def print_full_classification_report(y_true, y_pred, X_test=None, axes_names=None, classes_names=None):
    if axes_names is None:
        axes_names = ["x1", "x2"]
    
    assert len(axes_names) == 2, "axes_names params is not valid, you ahv eto give a list of 2 elements (or leave it None), because the plot is 2D"
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    if X_test is not None:
        print("Scatter plot of the test data:")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis")
        
        plt.xlabel(axes_names[0])
        plt.ylabel(axes_names[1])
        plt.title("Scatter plot of the test data")
        plt.colorbar()
        plt.show()

    print("Confusion matrix:")
    plot_confusion_matrix(y_true, y_pred, classes=classes_names)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes_names))

    # print("Silhouette Score:", silhouette_score(y_true, y_pred))


def print_full_regression_report(y_true, y_pred):
    # Calculate the mean absolute error
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean absolute error: {mae}")

    # Calculate the mean squared error
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean squared error: {mse}")

    # Calculate the relative error
    relative_errors = np.abs((y_true - y_pred) / y_true)
    mean_relative_error = np.mean(relative_errors)
    print(f"Mean relative error: {mean_relative_error}")


def create_training_history_plot(history):
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="valid")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    plt.plot(history.history["mae"], label="train")
    plt.plot(history.history["val_mae"], label="valid")
    plt.title("Model MAE")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    plt.plot(history.history["mse"], label="train")
    plt.plot(history.history["val_mse"], label="valid")
    plt.title("Model MSE")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
