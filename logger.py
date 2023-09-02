import logging

# Create a logger
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Set the logging level to INFO in production
logger.setLevel(logging.INFO)  

# Create a file handler and set the log file name
log_file = 'app.log'
file_handler = logging.FileHandler(log_file)

# Create a formatter and set the format for log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
