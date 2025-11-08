import matplotlib.pyplot as plt
import numpy as np

def plot(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, color="teal")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Nilai Prediksi")
    plt.title(f"Prediksi vs Aktual - {model_name}")
    plt.grid(True)
    plt.show()

def compare_models(results_dict):
    metrics = ["MAE", "RMSE", "R2"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    dt_values = [results_dict["Decision Tree"][m] for m in metrics]
    knn_values = [results_dict["KNN"][m] for m in metrics]

    ax.bar(x - width/2, dt_values, width, label="Decision Tree")
    ax.bar(x + width/2, knn_values, width, label="KNN")

    ax.set_xlabel("Metrik")
    ax.set_ylabel("Nilai")
    ax.set_title("Perbandingan Kinerja Model")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True)
    plt.show()
