from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan transformer
poly = joblib.load("models/poly.pkl")
scaler = joblib.load("models/scaler.pkl")
dt_model = joblib.load("models/best_dt.pkl")
knn_model = joblib.load("models/best_knn.pkl")

# Fungsi prediksi
def prediksi_negara(nama_negara):
    df = pd.read_csv("data/world-happiness-report-2021.csv")
    row = df[df["Country name"].str.lower() == nama_negara.lower()]

    if row.empty:
        return None

    features = [col for col in df.columns if col not in ["Country name", "Regional indicator", "Ladder score"]]
    data_full = row[features]
    data_poly = poly.transform(data_full)
    data_scaled = scaler.transform(data_poly)

    pred_dt = dt_model.predict(data_scaled)[0]
    pred_knn = knn_model.predict(data_scaled)[0]
    pred_avg = np.mean([pred_dt, pred_knn])

    actual = row["Ladder score"].values[0] if "Ladder score" in row.columns else None
    region = row["Regional indicator"].values[0]

    q1, q3 = df["Ladder score"].quantile([0.33, 0.66])
    if pred_avg >= q3:
        tingkat = "Sangat Bahagia"
    elif pred_avg >= q1:
        tingkat = "Bahagia"
    else:
        tingkat = "Kurang Bahagia"

    return {
        "country": nama_negara,
        "region": region,
        "pred_dt": round(pred_dt, 3),
        "pred_knn": round(pred_knn, 3),
        "pred_avg": round(pred_avg, 3),
        "actual": round(actual, 3) if actual else None,
        "category": tingkat,
    }

# Route utama
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        country = request.form["country"].strip()
        if not country:
            error = "Masukkan nama negara terlebih dahulu."
        else:
            result = prediksi_negara(country)
            if not result:
                error = f"Negara '{country}' tidak ditemukan."

    return render_template("index.html", result=result, error=error)


# Jalankan Flask
if __name__ == "__main__":
    app.run(debug=True)
