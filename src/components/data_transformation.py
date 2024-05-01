import pandas as pd
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = "artifacts/preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Started")
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

            # Define the custom ranking for each ordinal variable
            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = [
                "I1",
                "SI2",
                "SI1",
                "VS2",
                "VS1",
                "VVS2",
                "VVS1",
                "IF",
            ]

            logging.info("Pipleline initiated")

            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ("encoder", OrdinalEncoder()),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ]
            )
            logging.info("preprocessing completed")
            return preprocessor

        except Exception as e:
            logging.info("Error occured at DataTransformation stage")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initialization of Data Transformation started")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f"Train DataFrame Head : \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n{test_data.head().to_string()}")

            preprocessor = self.get_data_transformation()

            target_col = "price"
            drop_cols = [target_col, "id"]

            train_target = train_data[target_col]
            test_target = test_data[target_col]
            train_data = train_data.drop(columns=drop_cols, axis=1)
            test_data = test_data.drop(columns=drop_cols, axis=1)

            logging.info("Transforming data")
            train_data = preprocessor.fit_transform(train_data)
            test_data = preprocessor.transform(test_data)

            # Converting them to 2d array with target variable
            logging.info("reached")
            train_arr = np.c_[train_data, np.array(train_target)]
            test_arr = np.c_[test_data, np.array(test_target)]

            save_obj(
                preprocessor, self.data_transformation_config.preprocessor_obj_file_path
            )
            logging.info("preprocessor pickle file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Error occured at intializtion of Data Transformation")
            raise CustomException(e, sys)
