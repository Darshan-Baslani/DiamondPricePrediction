import numpy as np
import pickle
import os
import sys
from sklearn.metrics import r2_score

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.logger import logging
from src.exception import CustomException


def save_obj(obj, path):
    try:
        logging.info("Starting the process to save the object")
        dir_path = os.path.dirname(path)

        os.makedirs(dir_path, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        logging.info("Exception occured while saving the object")
        raise CustomException(e, sys)


i = 0


def model_evaluation(model, i, X_test, y_test, model_list):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    return r2
