from os import getenv
from dotenv import load_dotenv

from app.shared.logging_config import setup_logging

load_dotenv('../.env')
log_level = getenv('LOG_LEVEL', 'INFO').upper()
setup_logging(log_level)
