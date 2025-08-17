import logging
from os import getenv
from dotenv import load_dotenv

from fastapi import FastAPI
import uvicorn

from app.transcribers.routes import router
from app.shared.logging_config import setup_logging


load_dotenv('./app/transcribers/local_whisper/.env')
log_level = getenv('LOG_LEVEL', 'INFO').upper()
setup_logging(log_level)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

if __name__ == '__main__':
    logger.debug('Transcriber app started')
    uvicorn.run('main:app', port=3001)