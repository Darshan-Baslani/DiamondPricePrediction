import pandas as pd
import numpy as np
import os
import sys

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import model_evaluation, save_obj

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = "artifacts/model.pkl"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Starting Model Training")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1],
            )
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "BaggingRegressor": BaggingRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            i = 0
            model_list = [
                "LinearRegression",
                "Lasso",
                "Ridge",
                "ElasticNet",
                "GradientBoostingRegressor",
                "AdaBoostRegressor",
                "BaggingRegressor",
                "XGBRegressor",
            ]
            model_score = {}
            logging.info("Model Training is starting")
            for model in models:
                curr_model = models[model]
                curr_model.fit(X_train, y_train)
                model_score[model_list[i]] = model_evaluation(
                    curr_model, i, X_test, y_test, model_list
                )

                i += 1

            # To get best model score from dictionary
            best_model_score = max(sorted(model_score.values()))

            best_model_name = list(model_score.keys())[
                list(model_score.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best Model : {best_model}")
            print(f"Best Model : {best_model}")

            save_obj(best_model, self.model_trainer_config.trainer_model_file_path)
            logging.info("Model saved successfully")

        except Exception as e:
            logging.info("Error occured while training the model")
            raise CustomException(e, sys)
