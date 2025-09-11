# app/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from . import config

engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()