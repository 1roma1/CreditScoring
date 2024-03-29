import os
import pandas as pd
from sklearn.model_selection import train_test_split

import config


if __name__ == "__main__":
    df = (pd
        .read_csv(config.DATA_FILE_NAME)
        .drop(["client_id", "home_address", "work_address", "app_date"], axis=1)
        .dropna()
    )

    X, y = df.drop(["default"], axis=1), df['default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    train_df = pd.DataFrame(X_train, columns=X_train.columns)
    train_df = train_df.assign(default=y_train)

    test_df = pd.DataFrame(X_test, columns=X_test.columns)
    test_df = test_df.assign(default=y_test)

    train_df.to_csv(os.path.join(config.DATA_PATH, "train.csv"), index=False)
    test_df.to_csv(os.path.join(config.DATA_PATH, "test.csv"), index=False)
