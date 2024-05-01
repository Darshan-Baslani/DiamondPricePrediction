import pandas as pd
import os
import sys
from src.components import data_transformation

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initialize_ingestion()
    data_transformation = DataTransformation()
    (
        train_data,
        test_data,
        preprocessor_path,
    ) = data_transformation.initialize_data_transformation(
        train_data_path, test_data_path
    )

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_data, test_data)
