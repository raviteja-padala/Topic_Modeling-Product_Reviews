import logger

class CustomException(Exception):
    """Custom exception class for handling exceptions in the application."""

    def __init__(self, message):
        super().__init__(message)

def handle_exception(e):
    """Handle exceptions by logging and providing a custom error message."""
    error_message = str(e)
    # Log the error message
    logger.error(error_message)
    return "An error occurred: " + error_message, 500

