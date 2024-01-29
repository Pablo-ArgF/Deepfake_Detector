# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TrainingMetrics():
    def __init__(self, train_history) -> None:
        self.train_history = train_history
        self.history_dict = train_history.history
    
    def plot(self):
        loss_values = self.history_dict["loss"]
        val_loss_values = self.history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, loss_values, "bo", label="Pérdida de entrenamiento")
        plt.plot(epochs, val_loss_values, "b", label="Pérdida de validación")
        plt.title("Pérdida de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.show()
