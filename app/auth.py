import os
import bcrypt
from datetime import timedelta, datetime
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.models import User
from sqlmodel import Session, select
from app.db import engine


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"


def get_password_hash(plain_password: str):
    return bcrypt.hashpw(
        bytes(plain_password, encoding="utf-8"),
        bcrypt.gensalt(),
    ).decode("utf8")


def verify_password(plain_password, hashed_password) -> bool:
    """Verifies a hash against a password"""
    return bcrypt.checkpw(
        bytes(plain_password, encoding="utf-8"),
        bytes(hashed_password, encoding="utf-8"),
    )


def verify_token(token: str) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None

        with Session(engine) as session:
            user = session.exec(
                select(User).where(User.username == username)
            ).first()
            return user

    except JWTError:
        return None


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (
        expires_delta or timedelta(minutes=60 * 24 * 30)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_user_from_token(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    return user
