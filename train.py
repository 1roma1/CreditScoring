import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import \
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import config


def all_metrics(y_true, y_pred, y_pred_prob):
    temp_dict = {} 
    temp_dict['accuracy'] = [accuracy_score(y_true, y_pred)]
    temp_dict['balanced accuracy'] = [balanced_accuracy_score(y_true, y_pred)]
    temp_dict['precision'] = [precision_score(y_true, y_pred)]
    temp_dict['recall'] = [recall_score(y_true, y_pred)]
    temp_dict['f1_score'] = [f1_score(y_true, y_pred)]
    temp_dict['roc_auc'] = [roc_auc_score(y_true, y_pred_prob)]    

    return pd.DataFrame.from_dict(temp_dict)


def logreg_optimization(model, X_train, y_train):
    param_grid = [
        {'penalty': ['l1'], 
        'solver': ['liblinear', 'lbfgs'], 
        'class_weight': ['none', 'balanced'], 
        'multi_class': ['auto','ovr'], 
        'C': [0.1, 1, 10],
        'max_iter': [1000],
        'tol': [1e-5]},
        {'penalty': ['l2'], 
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
        'class_weight': ['none', 'balanced'], 
        'multi_class': ['auto','ovr'], 
        'C': [0.1, 1, 10], 
        'max_iter': [1000],
        'tol': [1e-5]},
        {'penalty': [None], 
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
        'class_weight':['none', 'balanced'], 
        'multi_class': ['auto','ovr'], 
        'C': [0.1, 1, 10], 
        'max_iter': [1000],
        'tol': [1e-5]}
    ]

    gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=5, verbose=3)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
    
    best_parameters = model.get_params()
    for param_name in sorted(best_parameters.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))

    return model


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, "test.csv"))

    num_columns = ["age", "decline_app_cnt", "score_bki", 
                   "bki_request_cnt", "region_rating", 
                   "income", "sna"]
    bin_columns = ["sex", "car", "car_type", "foreign_passport"]

    X_train, y_train = train_df.drop("default", axis=1), train_df["default"]
    X_test, y_test = test_df.drop("default", axis=1), test_df["default"]

    column_transformer = ColumnTransformer([
        ("bin", OrdinalEncoder(), bin_columns),
        ("num", StandardScaler(), num_columns),
        ("log_num", FunctionTransformer(np.log1p), ["age", "income"]),
        ("cat", OneHotEncoder(), ["education"])
    ])

    X_train_tr = column_transformer.fit_transform(X_train)

    model = LogisticRegression(random_state=config.RANDOM_STATE)
    model.fit(X_train_tr, y_train)

    print(model.__class__.__name__)
    print("_"*30)

    X_test_tr = column_transformer.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_tr)[:, 1]
    y_pred = model.predict(X_test_tr)
    print("Test:")
    print(all_metrics(y_test, y_pred, y_pred_proba))

    model = logreg_optimization(model, X_train_tr, y_train)

    preds = model.predict(X_test_tr)
    y_pred_proba = model.predict_proba(X_test_tr)[:, 1]
    y_pred = model.predict(X_test_tr)
    print("Test after optimizing:")
    print(all_metrics(y_test, y_pred, y_pred_proba))

    predict_pipeline = Pipeline([
        ("transformer", column_transformer), 
        ("predictor", model)
    ])

    joblib.dump(predict_pipeline, os.path.join(config.MODEL_PATH, "credit_scoring_model.pk"))
