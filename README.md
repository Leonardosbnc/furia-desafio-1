# 🤖 FuriaBot - Chatbot com FastAPI + RAG + OpenAI

Projeto FastAPI com autenticação JWT, WebSocket para um chatbot inteligente (FuriaBot), e integração com OpenAI utilizando RAG (Retrieval-Augmented Generation) para respostas baseadas em documentos `.json`.

---

## 🚀 Tecnologias Utilizadas

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLite](https://www.sqlite.org/)
- [OpenAI API](https://platform.openai.com/)
- WebSocket
- RAG (Retrieval-Augmented Generation)
- Docker (para deploy)

---

## 📁 Funcionalidades

- ✅ Login com JWT (`/login`)
- ✅ Criação de usuários (`/user`)
- ✅ Comunicação com o chatbot via WebSocket (`/ws/chat`)
- ✅ Indexação de arquivos `.json`
- ✅ Geração de respostas com RAG + OpenAI
- ✅ Deploy em servidor com Docker e Docker Compose (DigitalOcean)

---

## 🔐 Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
DATABASE_URL=sqlite:///./furia.db
SECRET_KEY=sua_chave_secreta
CORS_ORIGIN=http://localhost:3000  # ou domínio do frontend
OPENAI_API_KEY=sua_chave_openai
```

---

## 💻 Execução Local (com virtualenv)

1. Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute a aplicação:

```bash
uvicorn app.main:app --reload
```

> A API estará disponível em: `http://localhost:8000`

---

## 🔌 Endpoints Principais

| Método | Rota       | Descrição                                            |
| ------ | ---------- | ---------------------------------------------------- |
| POST   | `/login`   | Realiza login e retorna token JWT                    |
| POST   | `/user`    | Criação de novo usuário                              |
| WS     | `/ws/chat` | Comunicação em tempo real com FuriaBot via WebSocket |

---

## 🧠 Como funciona o FuriaBot?

1. Documentos `.json` são lidos e indexados.
2. O usuário envia perguntas via WebSocket.
3. O sistema busca os documentos mais relevantes.
4. O conteúdo é enviado à OpenAI para geração da resposta.
5. A resposta é retornada ao usuário em tempo real.

---

## ☁️ Deploy com Docker na DigitalOcean

1. Crie uma Droplet na DigitalOcean com Docker instalado.
2. Clone o repositório no servidor.
3. Configure o arquivo `.env` com as variáveis de produção.
4. Execute os comandos:

```bash
docker-compose up -d --build
```

> A aplicação estará disponível na porta 8000 da Droplet.
