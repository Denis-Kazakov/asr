import logging
from os import getenv
from dotenv import load_dotenv


from fastapi import FastAPI
import uvicorn

from app.nlp.transcript_segmentation.routes import router
from app.shared.logging_config import setup_logging


load_dotenv('./app/nlp/transcript_segmentation/stanza/.env')
log_level = getenv('LOG_LEVEL', 'INFO').upper()
setup_logging(log_level)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)


if __name__ == '__main__':
    logger.debug('Main app started')
    uvicorn.run('main:app')