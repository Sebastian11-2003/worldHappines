from src.data_preprocessing import preprocess_data
from src.model_training.tuning import tuning_decision_tree, tuning_knn
from src.model_evaluation import evaluasi_model
from src.visualization import plot, compare_models
from src.utils import simpan_model
from src.model_training.train import train_and_evaluate

if __name__ == "__main__":
    csv_path = "data/world-happiness-report-2021.csv"
    model_dt, model_knn, r2_train_dt, r2_test_dt, r2_train_knn, r2_test_knn = train_and_evaluate()
    
    # Preprocessing
    hasil = preprocess_data(csv_path)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = hasil[:7]
    y_train_class, y_test_class = hasil[7], hasil[8]

    # Hyperparameter tuning
    print("\n Melakukan tuning Decision Tree")
    best_dt = tuning_decision_tree(X_train, y_train)

    print("\n Melakukan tuning KNN")
    best_knn = tuning_knn(X_train_s, y_train)

    # Evaluasi model terbaik
    results = {}
    results["Decision Tree"] = evaluasi_model(best_dt, X_test, y_test, "Decision Tree Terbaik")
    results["KNN"] = evaluasi_model(best_knn, X_test_s, y_test, "KNN Terbaik")

    # Visualisasi hasil
    plot(y_test, best_dt.predict(X_test), "Decision Tree Terbaik")
    plot(y_test, best_knn.predict(X_test_s), "KNN Terbaik")
    compare_models(results)

    # Simpan model
    simpan_model(scaler, "models/scaler.pkl")
