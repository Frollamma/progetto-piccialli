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


def plot_confusion_matrix(y_true, y_pred):
    # Create the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="coolwarm", fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()


def print_full_classification_report(y_true, y_pred, target_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Confusion matrix:")
    plot_confusion_matrix(y_true, y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

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
