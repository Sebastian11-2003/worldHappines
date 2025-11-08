from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from src.data_preprocessing import preprocess_data
from src.utils import simpan_model
from src.model_training.tuning import tuning_decision_tree, tuning_knn

def latih_decision_tree(X_train, y_train, X_test, y_test):
    print("Training Decision Tree...")
    model_dt = tuning_decision_tree(X_train, y_train)
    model_dt.fit(X_train, y_train)

    from sklearn.model_selection import ShuffleSplit, cross_val_score
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(model_dt, X_train, y_train, scoring='r2', cv=cv)
    print(f"CV mean R² = {scores.mean():.4f} ± {scores.std():.4f}")


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_dt, X_train, y_train, cv=cv, scoring='r2')
    print(f"Cross-Validasi R²: Mean = {cv_scores.mean():.3f}, Std = {cv_scores.std():.3f}")

    y_pred = model_dt.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2_train = model_dt.score(X_train, y_train)
    r2_test = model_dt.score(X_test, y_test)
    print(f"R² Train: {r2_train:.3f} | R² Test: {r2_test:.3f}")
    print(f"Decision Tree — MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    

    simpan_model(model_dt, "models/best_dt.pkl")
    return model_dt, (mae, rmse, r2_train, r2_test)


def latih_knn(X_train_s, y_train, X_test_s, y_test):
    print("Training KNN...")
    model_knn = tuning_knn(X_train_s, y_train)
    model_knn.fit(X_train_s, y_train)

    from sklearn.model_selection import ShuffleSplit, cross_val_score
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(model_knn, X_train_s, y_train, scoring='r2', cv=cv) 
    print(f"CV mean R² = {scores.mean():.4f} ± {scores.std():.4f}")


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_knn, X_train_s, y_train, cv=cv, scoring='r2') 
    print(f"Cross-Validasi R²: Mean = {cv_scores.mean():.3f}, Std = {cv_scores.std():.3f}")

    y_pred = model_knn.predict(X_test_s) 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2_train = model_knn.score(X_train_s, y_train) 
    r2_test = model_knn.score(X_test_s, y_test) 
    print(f"R² Train: {r2_train:.3f} | R² Test: {r2_test:.3f}")
    print(f"KNN — MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    simpan_model(model_knn, "models/best_knn.pkl")
    return model_knn, (mae, rmse, r2_train, r2_test)


def train_and_evaluate():
    print("Memulai Training")
    csv_path = "data/world-happiness-report-2021.csv"
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler, y_train_class, y_test_class, poly = preprocess_data(csv_path)

    model_dt, metrics_dt = latih_decision_tree(X_train_s, y_train, X_test_s, y_test)
    model_knn, metrics_knn = latih_knn(X_train_s, y_train, X_test_s, y_test) 

    joblib.dump(poly, "models/poly.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Semua model & transformer berhasil disimpan di folder 'models/'")

    r2_train_dt, r2_test_dt = metrics_dt[2], metrics_dt[3]
    r2_train_knn, r2_test_knn = metrics_knn[2], metrics_knn[3]
    return model_dt, model_knn, r2_train_dt, r2_test_dt, r2_train_knn, r2_test_knn
