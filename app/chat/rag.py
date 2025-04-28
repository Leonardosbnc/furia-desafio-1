import os
from openai import OpenAI
from typing import List


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rag_query(user_query: str, docs: List[str], index, embedder, top_k=3):
    """Função de busca com FAISS para encontrar os documentos mais relevantes e gerar a resposta"""
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)

    # Busca pelos documentos mais relevantes
    _, I = index.search(
        query_embedding, top_k
    )  # I são os índices dos top_k documentos mais relevantes
    retrieved_docs = [
        docs[i] for i in I[0]
    ]  # Recupera os documentos correspondentes
    context = "\n\n".join(retrieved_docs)

    prompt = f"""Você é um assistente da torcida da FURIA. Responda com base nas informações abaixo.

                Informações:
                {context}

                Pergunta: {user_query}
                Resposta:"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # barato e eficaz
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente especializado na equipe de CS da FURIA.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )

    return completion.choices[0].message.content.strip()
