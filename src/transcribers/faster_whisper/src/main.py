import logging

from fastapi import FastAPI
import uvicorn

from logging_config import setup_logging
setup_logging()
from routes import router


logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

if __name__ == '__main__':
    logger.debug('Main app started')
    uvicorn.run('main:app')