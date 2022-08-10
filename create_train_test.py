import pandas as pd 
from sklearn.model_selection import train_test_split


def input_missing_values(df):
    df = df.fillna(df.median())
    return df 


def remove_outliers(df):
    df = df.drop(df[df['DebtRatio'] > 3489.025].index)
    df = df.drop(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index)
    df.loc[df['NumberOfTime30_59DaysPastDueNotWorse'] > 90, 'NumberOfTime30_59DaysPastDueNotWorse'] = 18
    df.loc[df['NumberOfTime60_89DaysPastDueNotWorse'] > 90, 'NumberOfTime60_89DaysPastDueNotWorse'] = 18
    df.loc[df['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18
    return df 


if __name__ == "__main__":
    df = pd.read_csv("dataset/dataset.csv")
    df = input_missing_values(df)
    df = remove_outliers(df)

    X = df.drop(['SeriousDlqin2yrs'], axis=1)
    y = df['SeriousDlqin2yrs']

    X_columns = X.columns
    y_column = ['SeriousDlqin2yrs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    train_df = pd.DataFrame(X_train, columns=X_columns)
    train_df = train_df.assign(SeriousDlqin2yrs=y_train)

    test_df = pd.DataFrame(X_test, columns=X_columns)
    test_df = test_df.assign(SeriousDlqin2yrs=y_test)

    train_df.to_csv("dataset/train.csv", index=False)
    test_df.to_csv("dataset/test.csv", index=False)
