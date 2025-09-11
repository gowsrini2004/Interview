# app/models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from datetime import datetime
from .database import Base
from . import config

class Video(Base):
    __tablename__ = "videos"
    id = Column(String(36), primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False)
    platform = Column(String(32), default="youtube")
    description = Column(Text)
    tags = Column(String(255))
    is_public = Column(Integer, default=1)
    added_by = Column(String(255), default="admin")
    access = Column(String(32), default="all")
    created_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String(16))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String(36), primary_key=True, index=True)
    uploader = Column(String(255))
    filename = Column(String(255))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class QuestionnaireAttempt(Base):
    __tablename__ = "q_attempts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    day = Column(String(10), index=True)
    score = Column(Float)
    comment = Column(Text)
    tips_json = Column(Text)
    session_id = Column(String(36))
    created_at = Column(DateTime, default=datetime.utcnow)

class QuestionnaireSession(Base):
    __tablename__ = "q_sessions"
    session_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    target_questions = Column(Integer, default=config.MIN_QUESTIONS)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)