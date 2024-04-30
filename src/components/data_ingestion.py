import pandas as pd
import sys
import os

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# Data ingestion Initialization
@dataclass
class DataIngestionConfig:
    # here we have used dataclass so we don't have to create a constructor
    # to initialize attributes
    train_data_path: str = "artifacts/train.csv"
    test_data_path: str = "artifacts/test.csv"
    raw_data_path: str = "artifacts/raw.csv"


# Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initialize_ingestion(self):
        logging.info("Data Ingestion Starts")
        try:
            df = pd.read_csv("notebook/data/gemstone.csv")
            logging.info("Data Read successfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw Data Created")

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=404)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Data Ingestion is Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info("Error occured at data ingestion Initialization")
            raise CustomException(e, sys)
