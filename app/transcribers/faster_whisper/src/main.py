import logging

from fastapi import FastAPI
import uvicorn

from app.utils.logging_config import setup_logging
setup_logging()
from app.routes import router


logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

if __name__ == '__main__':
    logger.debug('Transcriber app started')
    uvicorn.run('main:app')