import logging
from datetime import datetime
from pathlib import Path

# Define the log file path with start datetime
if Path.home().name == "nikolaibeckjensen":
    LOG_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet" / "Virk2Vec" / "data" / "logging"
elif Path.home().name == "annabramslow":
    LOG_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet(2)" / "Virk2Vec" / "data" / "logging"
elif Path.home().name == "147319":
    LOG_ROOT = Path.home() / "master_thesis" / "data" / "logging"
else: 
    LOG_ROOT = Path.home() / "data"


if Path.home().name == "nikolaibeckjensen":
    DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet" / "Virk2Vec" / "data"
elif Path.home().name == "annabramslow":
    DATA_ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-DanmarksTekniskeUniversitet(2)" / "Virk2Vec" / "data"
elif Path.home().name == "147319":
    DATA_ROOT = Path.home() / "master_thesis" / "data"
else: 
    DATA_ROOT = Path.home() / "data"


start_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = LOG_ROOT / f"serialize_{start_datetime}.log"

# Create a logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Set the logging level

# Create a file handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
log.addHandler(file_handler)

# Optionally, add a console handler if you want to see logs in the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)