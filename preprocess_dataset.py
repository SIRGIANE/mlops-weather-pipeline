import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils_and_constants import DROP_COLNAMES, PROCESSED_DATASET, RAW_DATASET, TARGET_COLUMN

def read_dataset(filename, drop_columns, target_column):
    df = pd.read_csv(filename).drop(columns=drop_columns)
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    return df

def target_encode_categorical_features(df, categorical_columns, target_column):
    encoded_data = df.copy()
    for col in categorical_columns:
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        encoded_data[col] = encoded_data[col].map(encoding_map)
    return encoded_data

def impute_and_scale_data(df_features):
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features.values)
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)
    return pd.DataFrame(X_preprocessed, columns=df_features.columns)

def main():
    weather = read_dataset(RAW_DATASET, DROP_COLNAMES, TARGET_COLUMN)
    categorical_columns = weather.select_dtypes(include=[object]).columns.to_list()
    weather = target_encode_categorical_features(weather, categorical_columns, TARGET_COLUMN)
    weather_features_processed = impute_and_scale_data(weather.drop(columns=TARGET_COLUMN))
    weather_labels = weather[TARGET_COLUMN]
    weather = pd.concat([weather_features_processed, weather_labels], axis=1)
    weather.to_csv(PROCESSED_DATASET, index=None)

if __name__ == "__main__":
    main()
