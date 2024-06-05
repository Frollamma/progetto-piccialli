# Print the confusion matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    silhouette_score,
    classification_report,
)
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
    print("Confusion matrix:")
    plot_confusion_matrix(y_true, y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Silhouette Score:", silhouette_score(y_true, y_pred))
