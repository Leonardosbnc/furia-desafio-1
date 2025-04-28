from dotenv import load_dotenv

load_dotenv()


import re
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    status,
    Depends,
    HTTPException,
)
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from app.auth import verify_token
from app.chat.documents import index, documents, embedder
from app.chat.rag import rag_query
from app.db import create_db_and_tables, get_session
from sqlmodel import Session, select
from app.models import User
from app.auth import get_password_hash, verify_password, create_access_token

from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


class UserRequest(BaseModel):
    username: str
    password: str


@app.post("/user", status_code=201)
def create_user(user_data: UserRequest, session: Session = Depends(get_session)):
    if re.fullmatch(r"^[A-Za-z]{3,16}$", user_data.username) is None:
        raise HTTPException(status_code=422, detail="Username Inválido")
    if len(user_data.password) < 6 or len(user_data.password) > 24:
        raise HTTPException(
            status_code=422, detail="Senha deve conter entre 6 e 24 caracteres"
        )

    user_exist = session.exec(
        select(User).where(User.username == user_data.username)
    ).first()
    if user_exist:
        raise HTTPException(status_code=422, detail="Usuário já existe")

    user = User(
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"message": "Usuário criado com sucesso"}


@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    user = session.exec(
        select(User).where(User.username == form_data.username)
    ).first()
    if not user or not verify_password(
        form_data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=401, detail="Usuário ou senha incorretos"
        )

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


# ==== WebSocket com Autenticação ====
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: str):
    print('hey')
    user = verify_token(token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        while True:
            question = await websocket.receive_text()
            answer = rag_query(question, documents, index, embedder)
            await websocket.send_text(answer)
    except WebSocketDisconnect:
        print(f"🔌 Conexão encerrada para {user}")
