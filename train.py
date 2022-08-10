import joblib
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(['SeriousDlqin2yrs'], axis=1)
    y = df['SeriousDlqin2yrs']
    return X, y


if __name__ == "__main__":

    X_train, y_train = load_data("dataset/train.csv")
    X_test, y_test = load_data("dataset/test.csv")

    num_cols = X_train.columns

    column_transformer = ColumnTransformer([
        ('num', StandardScaler(), num_cols), 
    ])

    X_train_tr = column_transformer.fit_transform(X_train)

    model = XGBClassifier(n_estimators = 120)

    model.fit(X_train_tr, y_train)

    print(model.__class__.__name__)
    print("_"*30)
    y_pred_proba = model.predict_proba(X_train_tr)[:, 1]
    y_pred = model.predict(X_train_tr)
    print("Train:")
    print(f"ROC AUC: {roc_auc_score(y_train, y_pred_proba)}")
    print(f"Precision: {precision_score(y_train, y_pred)}")
    print(f"Recall: {recall_score(y_train, y_pred)}")
    print(f"F1: {f1_score(y_train, y_pred)}")
    print("-"*30)

    X_test_tr = column_transformer.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_tr)[:, 1]
    y_pred = model.predict(X_test_tr)
    print("Test:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1: {f1_score(y_test, y_pred)}")

    predict_pipeline = Pipeline([
        ("transformer", column_transformer), 
        ("predictor", model)
    ])

    joblib.dump(predict_pipeline, "server/model/credit_scoring")
