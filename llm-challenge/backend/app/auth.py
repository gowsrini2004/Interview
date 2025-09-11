# app/auth.py

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from typing import Any

from . import config, models
from .database import get_db

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()

# --- Principal for Admin User ---
class AdminPrincipal:
    id = -1
    email = "admin"
    is_admin = True

def is_admin_principal(obj: Any) -> bool:
    return getattr(obj, "is_admin", False) is True

# --- Password and Token Utilities ---
def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    return pwd_ctx.verify(pw, hashed)

def create_access_token(sub: str, minutes: int = config.JWT_EXPIRE_MINUTES) -> str:
    exp = datetime.utcnow() + timedelta(minutes=minutes)
    return jwt.encode({"sub": sub, "exp": exp}, config.SECRET_KEY, algorithm="HS256")

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return sub
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- FastAPI Dependencies ---
def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)):
    token = creds.credentials
    sub = decode_token(token)
    if sub == "admin":
        return AdminPrincipal()
    
    user = db.query(models.User).filter(models.User.email == sub).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def require_admin(user=Depends(get_current_user)):
    if not is_admin_principal(user):
        raise HTTPException(status_code=403, detail="Admin only")
    return user