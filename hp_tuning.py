import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN

def load_data(path):
    data = pd.read_csv(path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y

def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)
    model = RandomForestClassifier()
    param_grid = json.load(open("hp_config.json", "r"))
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", json.dumps(best_params, indent=2))
    with open("rfc_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

if __name__ == "__main__":
    main()
