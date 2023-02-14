import os
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_PATH = os.path.join(ROOT_DIR, "data")
DATA_FILE_NAME = os.path.join(DATA_PATH, "data.csv")

MODEL_PATH = os.path.join(ROOT_DIR, "models")

RANDOM_STATE = 2
TEST_SIZE = 0.2