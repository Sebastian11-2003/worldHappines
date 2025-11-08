import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

def preprocess_data(csv_path,
                    features=None,
                    target='Ladder score',
                    test_size=0.2,
                    random_state=42,
                    poly_degree=1,
                    select_k=None,
                    clip_q_low=0.005,
                    clip_q_high=0.995):

    # Load dan bersihkan data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target]).drop_duplicates()

    # Tentukan fitur otomatis
    if features is None:
        features = [col for col in df.columns if col not in [target, 'Country name', 'Regional indicator']]

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Hitung batas data latih
    low_bounds = X_train.quantile(clip_q_low)
    high_bounds = X_train.quantile(clip_q_high)

    X_train = X_train.clip(lower=low_bounds, upper=high_bounds, axis=1)
    X_test = X_test.clip(lower=low_bounds, upper=high_bounds, axis=1)
    
    # PolynomialFeatures hanya di-fit pada train
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    X_train_poly = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out(features))
    X_test_poly = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(features))

    # Seleksi fitur opsional (hanya bila diperlukan)
    if select_k is not None and select_k < X_train_poly.shape[1]:
        selector = SelectKBest(score_func=f_regression, k=select_k)
        X_train_sel = selector.fit_transform(X_train_poly, y_train)
        X_test_sel = selector.transform(X_test_poly)
        X_train_poly = pd.DataFrame(X_train_sel, columns=X_train_poly.columns[selector.get_support()])
        X_test_poly = pd.DataFrame(X_test_sel, columns=X_test_poly.columns[selector.get_support()])

    # Standardisasi berdasarkan train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_poly)
    X_test_s = scaler.transform(X_test_poly)

    # Buat versi klasifikasi untuk eksperimen tambahan
    q1, q2, q3 = y.quantile([0.25, 0.5, 0.75])
    y_train_class = pd.cut(y_train, bins=[0, q1, q2, q3, y.max()],
                           labels=[0, 1, 2, 3], include_lowest=True)
    y_test_class = pd.cut(y_test, bins=[0, q1, q2, q3, y.max()],
                          labels=[0, 1, 2, 3], include_lowest=True)
    
    print(f"Q1 = {q1:.3f} | Median = {q2:.3f} | Q3 = {q3:.3f}")

    # Return lengkap agar kompatibel dengan main.py / file tuning
    return (
        X_train_poly, X_test_poly,
        y_train, y_test,
        X_train_s, X_test_s,
        scaler,
        y_train_class, y_test_class,
        poly
    )
