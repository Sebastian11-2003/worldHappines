from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluasi_model(model, X_test, y_test, nama_model):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    akurasi = r2 * 100  # hitung akurasi dari R²

    print(f"Evaluasi {nama_model}:")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"Akurasi Estimasi : {akurasi:.2f}%\n")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Akurasi (%)": akurasi
    }
