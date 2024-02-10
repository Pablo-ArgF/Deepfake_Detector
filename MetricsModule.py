# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class TrainingMetrics():
    def __init__(self, train_history) -> None:
        self.train_history = train_history
        self.history_dict = train_history.history
    
    def plot(self):
        loss_values = self.history_dict["loss"]
        val_loss_values = self.history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure()
        plt.plot(epochs, loss_values, "bo", label="Pérdida de entrenamiento (training loss)")
        plt.plot(epochs, val_loss_values, "b", label="Pérdida de validación (validation loss)")
        plt.title("Pérdida de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.show()
        return

    def confusionMatrix(self, real_y, predicted_y):
        # Create a confusion matrix
        conf_matrix = confusion_matrix(real_y, predicted_y)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        return

    def saveStats(self, filePath = "metrics.csv"):
        df = pd.DataFrame(self.history_dict)
        df.to_csv(filePath, index=True)
