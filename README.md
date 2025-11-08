# 🌍 World Happiness Prediction Web App (Flask + Machine Learning)

Aplikasi web ini dibuat untuk **memprediksi tingkat kebahagiaan suatu negara** berdasarkan data dari **World Happiness Report 2021**.  
Proyek ini menggunakan **Flask** sebagai backend dan dua algoritma Machine Learning yaitu **Decision Tree** dan **KNN** untuk melakukan prediksi.

---

## 📊 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis berbagai faktor sosial-ekonomi yang memengaruhi kebahagiaan masyarakat, serta memberikan **prediksi skor kebahagiaan** berdasarkan indikator yang ada pada dataset.

Faktor-faktor tersebut mencakup:
- GDP per Capita  
- Social Support  
- Healthy Life Expectancy  
- Freedom to Make Life Choices  
- Generosity  
- Perceptions of Corruption  

---

## ⚙️ Teknologi dan Tools yang Digunakan

- **Python 3.10+**
- **Flask** — backend web framework  
- **Pandas, NumPy** — data preprocessing  
- **Scikit-learn** — pelatihan dan tuning model ML  
- **Matplotlib, Seaborn** — visualisasi hasil  
- **Joblib** — penyimpanan model terlatih  
- **Render.com** — untuk deployment web app  


### Langkah Pemrosesan:
1. **Preprocessing Data**
   - Normalisasi fitur numerik
   - Pemisahan data training dan testing
   - Encoding label negara

2. **Hyperparameter Tuning**
   - Menggunakan `GridSearchCV` untuk mencari parameter optimal

3. **Evaluasi**
   - Menggunakan metrik: `MAE`, `RMSE`, dan `R² Score`

4. **Visualisasi**
   - Scatter plot antara nilai aktual dan prediksi
   - Perbandingan performa antar model


