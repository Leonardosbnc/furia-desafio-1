import os
import json
import numpy as np
import faiss
import re
import unidecode
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as nmlz


EMBEDDINGS_FILE = "data/embeddings/embeddings.npy"
INDEX_FILE = "data/embeddings/faiss_index.index"
BATCH_SIZE = 64
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r"[^\w\s\.,!?;:()\[\]\"'/-]", "", text)
    return text.strip()


def format_transfer(entry):
    acao = entry.get("acao")
    jogador = entry.get("jogador")
    data = entry.get("data")

    if acao == "transferência":
        return f"{jogador} foi transferido de {entry.get('origem')} para {entry.get('destino')} em {data}."
    elif acao == "entrada":
        return f"{jogador} entrou no time {entry.get('destino')} em {data}."
    elif acao == "banco":
        return f"{jogador} foi movido para o banco da {entry.get('time')} em {data}."
    elif acao == "saida":
        return f"{jogador} saiu da {entry.get('time')} em {data}."
    elif acao == "coach":
        return f"{jogador} assumiu o cargo de coach na {entry.get('time')} em {data}."
    else:
        return entry.get("descricao", "")


def format_match(entry):
    data = entry.get("data")
    desc = entry.get("descricao")
    adversario = entry.get("oponente")
    evento = entry.get("evento")
    resultado = entry.get("resultado")
    tipo = entry.get("bo")
    return f"{desc.split('o time')[0]} de {data[:4]}, a FURIA enfrentou {adversario} no evento {evento} e o resultado foi {resultado}. Foi uma série {tipo}."


def load_documents():
    print("Loading documents...")
    documents = []

    for file in [
        "data/source/_players.json",
        "data/source/team.json",
        "data/source/player_history.json",
    ]:
        with open(file, "r", encoding="utf-8") as f:
            print(f"Loading {file}")

            data = json.load(f)
            if isinstance(data, list):
                documents.extend(
                    normalize(json.dumps(item, ensure_ascii=False))
                    for item in data
                )
            elif isinstance(data, dict):
                for _, v in data.items():
                    documents.append(
                        normalize(". ".join(v) if isinstance(v, list) else v)
                    )

            print(f"Loaded {file}")
    for file in [
        "data/source/past_match_2017.json",
        "data/source/past_match_2018.json",
        "data/source/past_match_2019.json",
        "data/source/past_match_2020.json",
        "data/source/past_match_2021.json",
        "data/source/past_match_2022.json",
        "data/source/past_match_2023.json",
        "data/source/past_match_2024.json",
        "data/source/past_match_2025.json",
    ]:
        with open(file, "r", encoding="utf-8") as f:
            match_data = json.load(f)
            for item in match_data:
                documents.append(normalize(format_match(item)))

            print(f"Loaded {file}")
    with open("data/source/transfer_history.json", "r", encoding="utf-8") as f:
        print(f"Loading {file}")
        transfer_data = json.load(f)
        for item in transfer_data:
            documents.append(normalize(format_transfer(item)))

        print(f"Loaded {file}")
    with open("data/source/achievements.json", "r", encoding="utf-8") as f:
        print(f"Loading {file}")
        achievements_data = json.load(f)
        for item in achievements_data:
            documents.append(normalize(item["majors"]))
            for result in item["resultados"]:
                documents.append(normalize(result))

        print(f"Loaded {file}")

    return documents


def embed_documents(docs, embedder):
    print("Generating embeddings...")
    embeddings = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        batch_embeddings = embedder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    print("Embeddings generated")
    return np.vstack(embeddings)


def load_faiss_index(docs):
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if not os.path.exists("data/embeddings"):
        os.mkdir("data/embeddings")

    embeddings = embed_documents(docs, embedder)
    embeddings = nmlz(embeddings, axis=1)
    np.save(EMBEDDINGS_FILE, embeddings)

    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 4, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10
    faiss.write_index(index, INDEX_FILE)

    return index


documents = load_documents()
index = load_faiss_index(documents)
