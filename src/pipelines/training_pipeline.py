import pandas as pd
import os
import sys

sys.path.append("/media/darshan/Code/DiamondPricePrediction")
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initialize_ingestion()
    print(train_data_path)
