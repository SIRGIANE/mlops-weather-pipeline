from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=2, random_state=1993)
    model.fit(X_train, y_train)
    return model
