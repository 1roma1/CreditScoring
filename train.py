import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import config


def xgb_optimize(model, X, y, params=None):
    if params is None:
        params = {
            'n_estimators': np.arange(50, 200, 50),
            'max_depth': np.arange(3, 9, 2),
            'learning_rate': np.arange(0.05, 0.2, 0.05),
        }
    xgb_grid_search = GridSearchCV(model, params, scoring="roc_auc", cv=5, n_jobs=-1)
    xgb_grid_search.fit(X, y)
    return xgb_grid_search


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, "test.csv"))

    columns = ["age", "decline_app_cnt", "score_bki", 
               "bki_request_cnt", "region_rating", 
               "income", "sna"]

    X_train, y_train = train_df.drop("default", axis=1), train_df["default"]
    X_test, y_test = test_df.drop("default", axis=1), test_df["default"]

    column_transformer = ColumnTransformer([
        ('num', StandardScaler(), columns),
        ('cat', OneHotEncoder(), ["education"])
    ])

    X_train_tr = column_transformer.fit_transform(X_train)

    model = XGBClassifier()

    model.fit(X_train_tr, y_train)

    print(model.__class__.__name__)
    print("_"*30)
    y_pred_proba = model.predict_proba(X_train_tr)[:, 1]
    y_pred = model.predict(X_train_tr)
    print("Train:")
    print(f"ROC AUC: {roc_auc_score(y_train, y_pred_proba)}")
    print("-"*30)

    X_test_tr = column_transformer.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_tr)[:, 1]
    y_pred = model.predict(X_test_tr)
    print("Test:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba)}")

    model = xgb_optimize(model, X_train_tr, y_train)
    y_pred_proba = model.predict_proba(X_test_tr)[:, 1]
    y_pred = model.predict(X_test_tr)
    print("After optimization ROC AUC: ", roc_auc_score(y_test, y_pred_proba))    

    predict_pipeline = Pipeline([
        ("transformer", column_transformer), 
        ("predictor", model)
    ])

    joblib.dump(predict_pipeline, os.path.join(config.MODEL_PATH, "credit_scoring_model.pk"))
