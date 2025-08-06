import logging
from os import getenv
from dotenv import load_dotenv


from fastapi import FastAPI
import uvicorn

from app.transcribers.routes import router
from app.shared.logging_config import setup_logging


#  TODO. Log level is hard coded because it is not loading properly from .env. Fix it.
load_dotenv('../.env')
log_level = getenv('LOG_LEVEL', 'INFO').upper()
print('log level: ', log_level)
log_level = 'DEBUG'
setup_logging(log_level)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)


if __name__ == '__main__':
    logger.debug('Main app started')
    uvicorn.run('main:app')