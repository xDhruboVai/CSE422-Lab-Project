import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Function for StandardScaler
def scale_standard(df, features):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

# Function for MinMaxScaler
def scale_minmax(df, features):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

# Function for RobustScaler
def scale_robust(df, features):
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled


def correlation_plot(source):
    corr = source.corr(numeric_only = True)

    plt.figure(figsize = (12,8))
    sns.heatmap(corr, cmap = "YlGnBu", annot = True)
    plt.show()

def split(source, target = "NObeyesdad"):
    data_dir = "/Users/dihanislamdhrubo/Downloads/CSE422-Lab-Project-main/data/"    

    X = source.drop(target, axis=1)
    y = source[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,       # 30% test
        random_state=42,
        stratify=y           # preserves class distribution
    )

    X_train.to_csv(data_dir + "X_train.csv", index=False)
    X_test.to_csv(data_dir + "X_test.csv", index=False)
    y_train.to_csv(data_dir + "y_train.csv", index=False)
    y_test.to_csv(data_dir + "y_test.csv", index=False)

    print("Train set size:", X_train.shape)
    print("Test set size:", X_test.shape)

def outliers_plot(df):
    numeric_features = df.select_dtypes(include = ['int64', 'float64']).columns
    
    for col in numeric_features:
        plt.figure(figsize = (8, 4))
        sns.boxplot(x = df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

def outliers_summary(df):
    numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns

    print("Outlier Summary per Numeric Feature:\n")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        count = outliers.shape[0]

        print(f"{col}: {count} outlier(s)")
        if count > 0:
            print("Values:", outliers.values)
        print("-" * 50)

def scale_standard(df, features):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

def scale_minmax(df, features):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

def scale_robust(df, features):
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled