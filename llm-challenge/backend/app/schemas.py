# app/schemas.py

from pydantic import BaseModel, EmailStr, Field, AnyHttpUrl
from typing import List, Optional, Literal
from datetime import datetime

class VideoCreate(BaseModel):
    title: str = Field(..., max_length=255)
    url: AnyHttpUrl
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: bool = True

class VideoUpdate(BaseModel):
    title: Optional[str] = None
    url: Optional[AnyHttpUrl] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None

class VideoOut(BaseModel):
    id: str
    title: str
    url: str
    platform: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_public: bool
    added_by: Optional[str] = None
    access: str
    created_at: datetime

class RegisterIn(BaseModel):
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    identifier: str
    password: str

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
    num_predict: Optional[int] = 512
    provider: Literal["mistral", "groq"] = "mistral"

class ChatOut(BaseModel):
    reply: str
    sources: List[dict] = Field(default_factory=list)

class UpdateTitleIn(BaseModel):
    title: str

class QStartOut(BaseModel):
    session_id: str
    question: str

class QAnswerIn(BaseModel):
    session_id: str
    answer: str

class QAnswerOut(BaseModel):
    done: bool
    question: Optional[str] = None
    score: Optional[float] = None
    comment: Optional[str] = None
    tips: List[str] = Field(default_factory=list)

class QDashItem(BaseModel):
    day: str
    attempts: int
    avg_score: float
    latest_comment: Optional[str] = None