import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


EMBEDDINGS_FILE = "data/embeddings/embeddings.npy"
INDEX_FILE = "data/embeddings/faiss_index.index"
BATCH_SIZE = 64
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Recarrega documentos para consulta
def load_documents():
    print("Loading documents...")
    """Carrega e formata os documentos para geração de embeddings"""
    documents = []

    # Carregar dados de jogadores e equipe
    for file in [
        "data/source/_players.json",
        "data/source/team.json",
        "data/source/past_match.json",
        "data/source/transferencias.json",
        "data/source/historico_de_player.json",
    ]:
        with open(file, "r", encoding="utf-8") as f:
            print(f"Loading {file}")

            data = json.load(f)
            if isinstance(data, list):
                documents.extend(
                    json.dumps(item, ensure_ascii=False) for item in data
                )
            elif isinstance(data, dict):
                documents.append(json.dumps(data, ensure_ascii=False))

            print(f"Loaded {file}")

    return documents


def embed_documents(docs, embedder):
    """Gera embeddings para os documentos em lotes (batch)"""
    print("Generating embeddings...")
    embeddings = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_embeddings = embedder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    print("Embeddings generated")
    return np.vstack(embeddings)


def load_faiss_index(docs):
    """Cria ou carrega os embeddings e o índice FAISS"""
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        print("🔁 Carregando embeddings e índice FAISS do disco...")
        embeddings = np.load(EMBEDDINGS_FILE)
        index = faiss.read_index(INDEX_FILE)
    else:
        print("📌 Gerando novos embeddings...")
        embeddings = embed_documents(docs, embedder)
        np.save(EMBEDDINGS_FILE, embeddings)

        print("📌 Criando novo índice FAISS...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)

    return index


documents = load_documents()
index = load_faiss_index(documents)
