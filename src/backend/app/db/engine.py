from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from loguru import logger
import sys
from ..core.config import Settings

env_file = '/home/masoud/Desktop/projects/volleyball_analytics/conf/.env'
settings = Settings(_env_file=env_file, _env_file_encoding='utf-8')

logger.add(sink=sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message}", serialize=True)
db_uri = settings.get_db_uri()
mode = settings.MODE
engine = create_engine(db_uri)
logger.info(f"The `MODE` is {mode} ")
msg = f"The db = {settings.DEV_DB if mode != 'test' else settings.TEST_DB_URL} is selected."
logger.success(msg)


def get_db() -> Session:
    with Session(engine) as session:
        yield session


Base = declarative_base()
