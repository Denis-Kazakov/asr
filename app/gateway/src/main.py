import logging

from fastapi import FastAPI
import uvicorn

from routes import router


logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)

if __name__ == '__main__':
    logger.debug('Main app started')
    uvicorn.run('main:app')