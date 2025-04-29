import os
from openai import OpenAI
from typing import List


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rag_query(user_query: str, docs: List[str], index, embedder, top_k=3):
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)

    _, I = index.search(query_embedding, top_k)
    retrieved_docs = [docs[i] for i in I[0]]
    context = "\n\n".join(retrieved_docs)

    prompt = f"""Você é um assistente da torcida da FURIA. Responda com base nas informações abaixo.

                Informações:
                {context}

                Pergunta: {user_query}
                Resposta:"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
