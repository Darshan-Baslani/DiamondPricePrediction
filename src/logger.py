import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%s_%Y_%M_%H_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOGS_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format="[ %(asctiem)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)