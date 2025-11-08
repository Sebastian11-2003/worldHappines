from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Tuning Decision Tree
def tuning_decision_tree(X_train, y_train):
    grid = {
        'max_depth': [3, 5],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [5, 10,20],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Paraeter Decision Tree Terbaik:", grid.best_params_)
    print(f"R² terbaik: {grid.best_score_:.4f}")
    return grid.best_estimator_


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def tuning_knn(X, y):
    grid = {
        'n_neighbors': [15,20,30,40,50], 
        'weights': ['uniform'],
        'p': [2] 
    }

    knn = KNeighborsRegressor()
    
    grid_search = GridSearchCV(knn, 
                               grid, 
                               cv=5, 
                               scoring='r2', 
                               n_jobs=-1)
    
    grid_search.fit(X, y)

    print(f"Parameter KNN Terbaik: {grid_search.best_params_}")
    print(f"R² terbaik: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_